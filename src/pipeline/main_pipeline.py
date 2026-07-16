"""
Sentinel pipeline — tek hat VLM mimarisi:

1) High-res YOLO + MOG2 + track  (yalnızca algı / kırpma kararı)
2) Tek VLM @ 336×336, duvar saati ~2 sn periyot
   - Tetik yok  → full frame → 336²
   - Tetik var  → en büyük değişim + %20 dolgu → 336²
3) Yüksek çözünürlüklü Detail YOK · Gate/Detail ayrımı YOK

Model her zaman aynı boyutta beslenir; kırpılmış/kırpılmamış fark etmez.
Periyot: model çalışmasını bitirir, ardından 2 sn dolana kadar bekler.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Generator, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from src.decision.triage_engine import (
    CandidateBundle,
    TriggerKind,
    TriageDecision,
    TriageEngine,
)
from src.tracking.hybrid_tracker import FrameTracks, HybridTracker, TrackedObject
from src.vlm.internvl_agent import InternVLAgent, make_mock_generator
from src.vlm.schemas import AnalysisResult
from src.vlm.tools import ToolRegistry, ToolResult

logger = logging.getLogger("sentinel.pipeline")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

# Tek VLM girdisi
DEFAULT_VLM_SIZE = 336
DEFAULT_VLM_PERIOD_S = 2.0  # 0.5 Hz hedef; model bitmeden yeni istek yok
DEFAULT_BUFFER_SIZE = 30
DEFAULT_BURST_FRAMES = 3

# Geriye uyum alias isimleri
DEFAULT_GATE_SIZE = DEFAULT_VLM_SIZE
DEFAULT_GATE_HZ = 1.0 / DEFAULT_VLM_PERIOD_S  # 0.5


@dataclass
class FrameResult:
    frame_idx: int
    timestamp_s: float
    tracks: List[TrackedObject] = field(default_factory=list)
    triage: Optional[TriageDecision] = None
    candidates: Optional[CandidateBundle] = None
    gate: Optional[GateDecision] = None  # kullanılmıyor; geriye uyum
    gate_called: bool = False  # = vlm_called (geriye uyum)
    vlm_called: bool = False
    analysis: Optional[AnalysisResult] = None
    tool_results: List[ToolResult] = field(default_factory=list)
    report_path: Optional[str] = None
    high_res_tracker: bool = True
    gate_low_res_size: Optional[Tuple[int, int]] = None  # = vlm 336
    detail_high_res_shape: Optional[Tuple[int, ...]] = None  # artık hep 336
    vlm_input_shape: Optional[Tuple[int, ...]] = None
    cropped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp_s": self.timestamp_s,
            "track_ids": [t.track_id for t in self.tracks],
            "gate_called": self.gate_called,
            "vlm_called": self.vlm_called,
            "cropped": self.cropped,
            "vlm_input_shape": self.vlm_input_shape,
            "gate": self.gate.model_dump() if self.gate else None,
            "triage": self.triage.to_dict() if self.triage else None,
            "analysis": self.analysis.model_dump() if self.analysis else None,
            "tools": [r.to_dict() for r in self.tool_results],
            "report_path": self.report_path,
        }


@dataclass
class PipelineResult:
    source: str
    frames_processed: int = 0
    gate_calls: int = 0  # geriye uyum = vlm_calls
    vlm_calls: int = 0
    reports: List[str] = field(default_factory=list)
    frame_results: List[FrameResult] = field(default_factory=list)
    last_analysis: Optional[AnalysisResult] = None
    last_summary: str = ""
    last_gate: Optional[GateDecision] = None
    fps: float = 30.0
    duration_s: float = 0.0
    stats: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "frames_processed": self.frames_processed,
            "gate_calls": self.gate_calls,
            "vlm_calls": self.vlm_calls,
            "reports": self.reports,
            "last_summary": self.last_summary,
            "last_analysis": self.last_analysis.model_dump() if self.last_analysis else None,
            "fps": self.fps,
            "duration_s": self.duration_s,
            "stats": self.stats,
            "success": self.success,
            "error": self.error,
            "frame_results_count": len(self.frame_results),
        }


def downscale_for_vlm(frame: np.ndarray, size: int = DEFAULT_VLM_SIZE) -> np.ndarray:
    """Tek VLM girdisi: her zaman size×size."""
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


# Geriye uyum
def downscale_for_gate(frame: np.ndarray, size: int = DEFAULT_VLM_SIZE) -> np.ndarray:
    return downscale_for_vlm(frame, size)


def get_motion_crop_box(
    frame: np.ndarray,
    mog2_mask: Optional[np.ndarray],
    min_area: int = 300,
    padding_pct: float = 0.2,
) -> Optional[Tuple[int, int, int, int]]:
    """MOG2 maskesinden en büyük hareket alanı + padding → (x, y, w, h)."""
    if mog2_mask is None or mog2_mask.size == 0:
        return None
    try:
        contours, _ = cv2.findContours(
            mog2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        valid = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= min_area:
                valid.append((area, c))
        if not valid:
            return None

        _, max_cnt = max(valid, key=lambda x: x[0])
        x, y, w, h = cv2.boundingRect(max_cnt)
        h_img, w_img = frame.shape[:2]
        px = int(w * padding_pct)
        py = int(h * padding_pct)
        x1 = max(0, x - px)
        y1 = max(0, y - py)
        x2 = min(w_img, x + w + px)
        y2 = min(h_img, y + h + py)
        return (x1, y1, x2 - x1, y2 - y1)
    except Exception:
        return None


def crop_motion_patch(frame: np.ndarray, crop_box: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = crop_box
    return frame[y : y + h, x : x + w].copy()


def tracks_union_box(
    tracks: Sequence[TrackedObject],
    frame_shape: Tuple[int, ...],
    padding_pct: float = 0.2,
) -> Optional[Tuple[int, int, int, int]]:
    """YOLO/MOG track bbox birleşimi + padding."""
    if not tracks:
        return None
    h_img = int(frame_shape[0])
    w_img = int(frame_shape[1])
    xs1, ys1, xs2, ys2 = [], [], [], []
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        xs1.append(int(x1))
        ys1.append(int(y1))
        xs2.append(int(x2))
        ys2.append(int(y2))
    x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    px, py = int(w * padding_pct), int(h * padding_pct)
    xa = max(0, x1 - px)
    ya = max(0, y1 - py)
    xb = min(w_img, x2 + px)
    yb = min(h_img, y2 + py)
    return (xa, ya, xb - xa, yb - ya)


class SentinelPipeline:
    def __init__(
        self,
        tracker: Optional[HybridTracker] = None,
        triage: Optional[TriageEngine] = None,
        agent: Optional[InternVLAgent] = None,
        tools: Optional[ToolRegistry] = None,
        report_dir: Optional[Union[str, Path]] = None,
        roi_polygon: Optional[Sequence[Tuple[float, float]]] = None,
        save_reports: bool = True,
        store_frame_results: bool = True,
        max_frames: Optional[int] = None,
        time_fn: Optional[Callable[[], float]] = None,
        vlm_size: int = DEFAULT_VLM_SIZE,
        vlm_period_s: float = DEFAULT_VLM_PERIOD_S,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        burst_frames: int = DEFAULT_BURST_FRAMES,
        # Geriye uyum parametreleri
        gate_size: Optional[int] = None,
        gate_hz: Optional[float] = None,
        force_gate_every_candidate: bool = False,
        force_scan_every_frame: Optional[bool] = None,
        low_res_size: Optional[int] = None,
        scan_hz: Optional[float] = None,
    ) -> None:
        if force_scan_every_frame is not None:
            force_gate_every_candidate = force_scan_every_frame
        if low_res_size is not None:
            vlm_size = low_res_size
        if gate_size is not None:
            vlm_size = gate_size
        if scan_hz is not None and scan_hz > 0:
            vlm_period_s = 1.0 / float(scan_hz)
        if gate_hz is not None and gate_hz > 0:
            vlm_period_s = 1.0 / float(gate_hz)

        self.report_dir = Path(report_dir) if report_dir else DEFAULT_REPORT_DIR
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.save_reports = save_reports
        self.store_frame_results = store_frame_results
        self.max_frames = max_frames
        self._time_fn = time_fn or time.monotonic

        self.vlm_size = int(vlm_size)
        self.vlm_period_s = float(vlm_period_s) if vlm_period_s > 0 else DEFAULT_VLM_PERIOD_S
        self.buffer_size = int(buffer_size)
        self.burst_frames = int(burst_frames)
        self.force_gate_every_candidate = force_gate_every_candidate
        self.force_vlm_every_frame = force_gate_every_candidate

        # Geriye uyum
        self.gate_size = self.vlm_size
        self.gate_hz = 1.0 / self.vlm_period_s
        self.low_res_size = self.vlm_size
        self.scan_hz = self.gate_hz
        self.force_scan_every_frame = force_gate_every_candidate

        self.tools = tools or ToolRegistry(report_dir=self.report_dir)
        self.tracker = tracker
        if triage is not None:
            self.triage = triage
        else:
            self.triage = TriageEngine(
                roi_polygon=list(roi_polygon) if roi_polygon else None,
                time_fn=self._time_fn,
            )
        if agent is not None:
            self.agent = agent
            if tools is not None:
                self.agent.tools = self.tools
        else:
            self.agent = None

        self._frame_idx = 0
        self._frame_buffer: Deque[Tuple[int, np.ndarray]] = deque(maxlen=self.buffer_size)
        self._last_vlm_ts: Optional[float] = None  # senkron zaman damgası
        self.last_frame_result: Optional[FrameResult] = None
        self.last_pipeline_result: Optional[PipelineResult] = None

        # Tek VLM durumu
        self._vlm_busy = False
        self._vlm_call_count = 0
        self._vlm_status: str = "idle"  # idle | analyzing | danger_found | no_danger | error
        self._vlm_last_risk: str = ""
        self._last_vlm_summary = "Henüz VLM çalışmadı."
        self._last_vlm_json = "{}"
        self._last_vlm_called_on_frame = -1
        self._last_cropped = False
        self._vlm_triggered_until_frame = 0
        self._last_vlm_error: str = ""
        self._next_vlm_not_before: Optional[float] = None  # duvar saati
        self._vlm_cycle_start: Optional[float] = None

        # Ölü kod (Gate) temizlendi

        # Snapshot + worker / one-shot
        self._snap_lock = threading.Lock()
        self._snapshot: Optional[Dict[str, Any]] = None
        self._vlm_worker_stop = threading.Event()
        self._vlm_worker_thread: Optional[threading.Thread] = None
        self._vlm_schedule_enabled = False  # True iken process_frame tetikler
        self._vlm_claim_lock = threading.Lock()



    def ensure_components(self) -> None:
        if self.tracker is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tracker = HybridTracker(device=device, use_ultralytics_botsort=False)
        if self.agent is None:
            self.agent = InternVLAgent(
                tools=self.tools,
                generator_fn=make_mock_generator(),
                auto_execute_tools=True,
            )

    def reset(self) -> None:
        self.stop_vlm_worker()
        self._frame_idx = 0
        self._frame_buffer.clear()
        self._last_vlm_ts = None
        self.last_frame_result = None
        self._vlm_busy = False
        self._vlm_call_count = 0
        self._vlm_status = "idle"
        self._vlm_last_risk = ""
        self._last_vlm_summary = "Henüz VLM çalışmadı."
        self._last_vlm_json = "{}"
        self._last_vlm_called_on_frame = -1
        self._last_cropped = False
        self._vlm_triggered_until_frame = 0
        self._last_vlm_error = ""
        self._next_vlm_not_before = None
        self._vlm_cycle_start = None
        self._vlm_schedule_enabled = False
        with self._snap_lock:
            self._snapshot = None
        if self.tracker is not None and hasattr(self.tracker, "reset"):
            self.tracker.reset()
        if self.triage is not None:
            self.triage.reset()
        if self.agent is not None:
            self.agent.reset_memory()
            if self.agent.tools is not None:
                self.agent.tools.reset_history()
        self.tools.reset_history()

    def _vlm_period_allowed(self, ts: float) -> bool:
        if self.force_vlm_every_frame:
            return True
        if self._last_vlm_ts is None:
            return True
        return (ts - self._last_vlm_ts) >= self.vlm_period_s - 1e-9

    # Geriye uyum
    def _gate_allowed(self, ts: float) -> bool:
        return self._vlm_period_allowed(ts)

    # ------------------------------------------------------------------
    # Snapshot + tetik → 336 girdisi
    # ------------------------------------------------------------------
    def _should_crop(self, decision: TriageDecision, crop_box: Optional[Tuple]) -> bool:
        """YOLO/MOG2/aday tetik varsa kırp (pure yoksa full)."""
        if crop_box is None:
            return False
        if decision.bypass_to_detail:
            return True
        if decision.should_call_vlm:
            kinds = {t.kind for t in decision.triggers}
            if kinds and kinds <= {TriggerKind.PERIODIC}:
                return False
            return True
        # Soft sinyal: triage tetik listesi (cooldown yüzünden should_call false olsa bile)
        if decision.triggers:
            kinds = {t.kind for t in decision.triggers}
            if kinds - {TriggerKind.PERIODIC}:
                return True
        return False

    def _build_vlm_input(
        self,
        frame: np.ndarray,
        decision: TriageDecision,
        tracks: List[TrackedObject],
    ) -> Tuple[np.ndarray, bool, str]:
        """
        Returns: (336x336 BGR, cropped?, prompt_suffix)
        """
        mog2_mask = getattr(getattr(self.tracker, "mog2", None), "last_cleaned_mask", None)
        crop_box = get_motion_crop_box(frame, mog2_mask, min_area=250, padding_pct=0.2)
        if crop_box is None:
            crop_box = tracks_union_box(tracks, frame.shape, padding_pct=0.2)

        do_crop = self._should_crop(decision, crop_box)
        if do_crop and crop_box is not None:
            _, _, cw, ch = crop_box
            if cw >= 2 and ch >= 2:
                patch = crop_motion_patch(frame, crop_box)
                if patch is not None and getattr(patch, "size", 0) > 0:
                    low = downscale_for_vlm(patch, self.vlm_size)
                    suffix = (
                        "\n[ODAK]: Hareket/aday bölgesi %20 dolgu ile kırpıldı; "
                        "girdi yine 336×336 (yüksek çözünürlük yok)."
                    )
                    return low, True, suffix

        low = downscale_for_vlm(frame, self.vlm_size)
        return low, False, ""

    def _publish_snapshot(
        self,
        vlm_bgr_336: np.ndarray,
        frame_idx: int,
        fps: float,
        tracks: List[TrackedObject],
        trigger_info: str,
        cropped: bool,
        mode: str = "incident",
    ) -> None:
        with self._snap_lock:
            self._snapshot = {
                "image": vlm_bgr_336.copy(),
                "frame_idx": frame_idx,
                "fps": fps,
                "tracks": list(tracks),
                "trigger_info": trigger_info,
                "cropped": cropped,
                "mode": mode,
            }

    # ------------------------------------------------------------------
    # VLM worker — 2 sn periyot, model bitsin
    # ------------------------------------------------------------------
    def start_vlm_worker(self) -> None:
        """
        UI: VLM zamanlamasını aç.
        Asıl tetik process_frame(run_async) içinden (kare geldiğinde).
        Ek olarak yedek poller: snap var / kare yavaşsa bile kaçırma.
        """
        if self.tracker is None or self.agent is None:
            self.ensure_components()
        # Model yüklü mü?
        if self.agent is not None and not self.agent.is_loaded:
            try:
                self.agent.load()
            except Exception as e:
                self._last_vlm_error = f"Model yüklenemedi: {e}"
                logger.error(self._last_vlm_error)

        self._vlm_schedule_enabled = True
        self._vlm_worker_stop.clear()
        self._next_vlm_not_before = None  # hemen ilk denemeye izin

        if self._vlm_worker_thread is not None and self._vlm_worker_thread.is_alive():
            return
        self._vlm_worker_thread = threading.Thread(
            target=self._vlm_worker_loop,
            name="sentinel-vlm-poll",
            daemon=True,
        )
        self._vlm_worker_thread.start()
        logger.info(
            "VLM zamanlama açık: period=%.1fs size=%d (model bitmeden yenisi yok)",
            self.vlm_period_s,
            self.vlm_size,
        )

    def stop_vlm_worker(self) -> None:
        self._vlm_schedule_enabled = False
        self._vlm_worker_stop.set()
        t = self._vlm_worker_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            # Bitmekte olan çıkarımı kesme — uzun bekle
            t.join(timeout=max(5.0, self.vlm_period_s + 2.0))
        self._vlm_worker_thread = None

    def wait_vlm_idle(self, timeout_s: float = 120.0) -> None:
        """Mevcut VLM bitene kadar bekle (video sonu)."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        while self._vlm_busy and time.monotonic() < deadline:
            time.sleep(0.1)

    def ensure_at_least_one_vlm(self) -> bool:
        """
        Video bitti, hâlâ 0 çağrı varsa senkron 1 kez çalıştır.
        (Worker kaçırdıysa / model yavaş başladıysa kullanıcı 0 görmesin.)
        """
        if self._vlm_call_count > 0:
            return True
        with self._snap_lock:
            if self._snapshot is None:
                return False
        self.wait_vlm_idle(timeout_s=5.0)
        if self._vlm_call_count > 0:
            return True
        if self._vlm_busy:
            return False
        logger.warning("VLM hâlâ 0 — senkron zorunlu 1 çağrı")
        return self._run_one_vlm_cycle(blocking_claim=True)

    # Geriye uyum isimler
    def start_gate_worker(self) -> None:
        self.start_vlm_worker()

    def stop_gate_worker(self) -> None:
        self.stop_vlm_worker()

    def schedule_vlm_if_due(self) -> bool:
        """
        Geriye uyum: sürekli worker zaten poll eder.
        process_frame sadece snapshot yayınlar; bu no-op True döner.
        """
        return self._vlm_schedule_enabled and self._snapshot is not None

    def _vlm_worker_loop(self) -> None:
        """
        Sürekli VLM döngüsü (baştan sona):
          snapshot bekle → period doluysa bu thread'de çıkarım (bitene kadar) → tekrarla.
        One-shot thread yok; model durmadan yeni kare snapshot'ına bakar.
        """
        logger.info("VLM sürekli worker döngüsü başladı")
        while not self._vlm_worker_stop.is_set():
            if not self._vlm_schedule_enabled or self.agent is None:
                if self._vlm_worker_stop.wait(0.15):
                    break
                continue

            with self._snap_lock:
                has_snap = self._snapshot is not None
            if not has_snap:
                if self._vlm_worker_stop.wait(0.05):
                    break
                continue

            now = time.monotonic()
            if (
                self._next_vlm_not_before is not None
                and now < self._next_vlm_not_before - 1e-6
            ):
                wait = min(0.1, self._next_vlm_not_before - now)
                if self._vlm_worker_stop.wait(max(0.01, wait)):
                    break
                continue

            with self._vlm_claim_lock:
                if self._vlm_busy:
                    if self._vlm_worker_stop.wait(0.05):
                        break
                    continue
                self._vlm_busy = True
                self._gate_busy = True
                self._detail_busy = True
                self._vlm_status = "analyzing"
                self._detail_status = "analyzing"
                self._gate_status = "analyzing"
                self._vlm_cycle_start = time.monotonic()
                self._last_vlm_error = ""

            try:
                # Bu thread'de senkron çalış — bitmeden yenisi yok, sürekli sıradaki kare
                self._execute_vlm_from_snapshot()
            except Exception as e:
                try:
                    from src.vlm.cuda_compat import humanize_cuda_error

                    msg = humanize_cuda_error(e)
                except Exception:
                    msg = str(e)
                logger.error("VLM worker döngü hatası: %s", msg)
                self._last_vlm_error = msg
                self._vlm_status = "error"
            finally:
                end = time.monotonic()
                start = self._vlm_cycle_start or end
                # En az period aralığı; model yavaşsa hemen sonraki tura geç
                self._next_vlm_not_before = max(end, start + self.vlm_period_s)
                self._vlm_busy = False
                self._gate_busy = False
                self._detail_busy = False
                if self._vlm_status == "analyzing":
                    self._vlm_status = "idle"
                self._gate_status = "idle"

        logger.info("VLM sürekli worker döngüsü bitti")

    def _run_one_vlm_cycle(self, blocking_claim: bool = False) -> bool:
        """Senkron tek tur (ensure_at_least_one / test)."""
        if self.agent is None:
            return False
        with self._snap_lock:
            if self._snapshot is None:
                return False
        # Atomik claim — worker ile aynı lock kullanılır (race condition önleme)
        with self._vlm_claim_lock:
            if self._vlm_busy and not blocking_claim:
                return False
            self._vlm_busy = True
            self._gate_busy = True
            self._detail_busy = True
            self._vlm_status = "analyzing"
            self._vlm_cycle_start = time.monotonic()
            self._last_vlm_error = ""
        start = self._vlm_cycle_start
        try:
            return self._execute_vlm_from_snapshot()
        finally:
            end = time.monotonic()
            self._next_vlm_not_before = max(end, start + self.vlm_period_s)
            self._vlm_busy = False
            self._gate_busy = False
            self._detail_busy = False
            self._gate_status = "idle"

    def _execute_vlm_from_snapshot(self) -> bool:
        """Snapshot'tan tek analyze_detail. busy zaten claim edilmiş olmalı."""
        if self.agent is None:
            return False
        with self._snap_lock:
            snap = self._snapshot
            if snap is None:
                return False
            image = snap["image"]
            frame_idx = snap["frame_idx"]
            fps = snap["fps"]
            tracks = list(snap["tracks"])
            trigger_info = snap["trigger_info"]
            cropped = bool(snap["cropped"])
            mode = snap.get("mode", "incident")

        try:
            # Gerçek model yüklü değilse dene
            if (
                getattr(self.agent, "backend", "mock") != "mock"
                and getattr(self.agent, "generator_fn", None) is None
                and not self.agent.is_loaded
            ):
                self.agent.load()

            analysis = self.agent.analyze_detail(
                frame=image,
                frame_idx=frame_idx,
                fps=fps,
                tracks=tracks,
                trigger_info=trigger_info,
                execute_tools=True,
                mode=mode,
            )
            self._vlm_call_count += 1
            
            self._last_vlm_called_on_frame = frame_idx
            self._last_cropped = cropped
            self._vlm_triggered_until_frame = frame_idx + 30
            self._last_vlm_error = ""

            if mode == "routine":
                self._last_vlm_summary = analysis.log
                # Rutin logları JSON paneline yansıtma (kullanıcı sadece metin görmek istiyor)
                # self._last_vlm_json = json.dumps({"log": analysis.log, "is_danger": analysis.is_danger}, ensure_ascii=False)
                self._vlm_last_risk = "Düşük"
                self._detail_last_risk = "Düşük"
                
                if analysis.is_danger:
                    self._vlm_status = "danger_found"
                    self._force_next_incident = True
                else:
                    self._vlm_status = "no_danger"
                    
                logger.info(
                    "VLM(Routine) #%s frame=%s log=%r is_danger=%s",
                    self._vlm_call_count,
                    frame_idx,
                    analysis.log,
                    analysis.is_danger,
                )
                return True
            else:
                self._last_vlm_summary = analysis.summary
                self._last_vlm_json = json.dumps(
                    analysis.model_dump(), ensure_ascii=False, indent=2
                )
                self._vlm_last_risk = analysis.risk or ""
                self._detail_last_risk = self._vlm_last_risk

                risk_lower = (analysis.risk or "").lower()
                if risk_lower in ("yüksek", "kritik", "high", "critical") or (
                    analysis.events and len(analysis.events) > 0
                ):
                    self._vlm_status = "danger_found"
                else:
                    self._vlm_status = "no_danger"

                self.triage.notify_vlm_result(
                    risk=analysis.risk,
                    track_ids=[t.track_id for t in tracks],
                    timestamp=time.monotonic(),
                )
                if self.save_reports:
                    fr = FrameResult(
                        frame_idx=frame_idx,
                        timestamp_s=frame_idx / max(fps, 1e-6),
                        tracks=tracks,
                        vlm_called=True,
                        analysis=analysis,
                        vlm_input_shape=tuple(image.shape),
                        detail_high_res_shape=tuple(image.shape),
                        gate_low_res_size=(self.vlm_size, self.vlm_size),
                        cropped=cropped,
                        tool_results=list(self.agent.last_tool_results),
                    )
                    self._save_report(analysis, fr, fps=fps)

                logger.info(
                    "VLM(Incident) #%s frame=%s cropped=%s risk=%s size=%s",
                    self._vlm_call_count,
                    frame_idx,
                    cropped,
                    analysis.risk,
                    image.shape[:2],
                )
                return True
        except Exception as e:
            try:
                from src.vlm.cuda_compat import humanize_cuda_error

                msg = humanize_cuda_error(e)
            except Exception:
                msg = f"{type(e).__name__}: {e}"
            logger.error("VLM hatası: %s", msg)
            self._last_vlm_error = msg
            self._vlm_status = "error"
            self._detail_status = "no_danger"
            return False

    def _run_vlm_sync(
        self,
        result: FrameResult,
        image_336: np.ndarray,
        tracks: List[TrackedObject],
        idx: int,
        fps: float,
        ts: float,
        trigger: str,
        cropped: bool,
        mode: str = "incident",
    ) -> None:
        """Senkron test/benchmark yolu — tek 336 girdi, high-res yok."""
        analysis = self.agent.analyze_detail(
            frame=image_336,
            frame_idx=idx,
            fps=fps,
            tracks=tracks,
            trigger_info=trigger,
            execute_tools=True,
            mode=mode,
        )
        result.vlm_called = True
        result.gate_called = True
        result.vlm_input_shape = tuple(image_336.shape)
        result.detail_high_res_shape = tuple(image_336.shape)
        result.gate_low_res_size = (self.vlm_size, self.vlm_size)
        result.cropped = cropped
        self._vlm_call_count += 1
        self._last_vlm_ts = ts
        self._last_vlm_called_on_frame = idx
        self._last_cropped = cropped
        
        if mode == "routine":
            result.analysis = None
            self._last_vlm_summary = analysis.log
            self._last_vlm_json = json.dumps({"log": analysis.log, "is_danger": analysis.is_danger}, ensure_ascii=False)
            self._vlm_last_risk = "Düşük"
            if analysis.is_danger:
                self._force_next_incident = True
        else:
            result.analysis = analysis
            result.tool_results = list(self.agent.last_tool_results)
            self._last_vlm_summary = analysis.summary
            self._last_vlm_json = json.dumps(
                analysis.model_dump(), ensure_ascii=False, indent=2
            )
            self._vlm_last_risk = analysis.risk or ""
            self.triage.notify_vlm_result(
                risk=analysis.risk,
                track_ids=[t.track_id for t in tracks],
                timestamp=ts,
            )
            if self.save_reports:
                path = self._save_report(analysis, result, fps=fps)
                result.report_path = str(path)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: Optional[int] = None,
        fps: float = 30.0,
        timestamp_s: Optional[float] = None,
        yolo_detections: Optional[np.ndarray] = None,
        force_scan: bool = False,
        run_async: bool = False,
    ) -> FrameResult:
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("Boş kare")

        if self.tracker is None or self.agent is None:
            self.ensure_components()

        idx = self._frame_idx if frame_idx is None else int(frame_idx)
        if frame_idx is None:
            self._frame_idx += 1
        else:
            self._frame_idx = idx + 1

        ts = self._time_fn() if timestamp_s is None else float(timestamp_s)
        video_time = idx / max(fps, 1e-6)

        self._frame_buffer.append((idx, frame.copy()))

        # Kademe 1: high-res track (VLM değil)
        if yolo_detections is not None:
            ft: FrameTracks = self.tracker.process_frame(
                frame, yolo_detections=yolo_detections, run_mog2=True
            )
        else:
            ft = self.tracker.process_frame(frame, run_mog2=True)
        tracks = list(ft.tracks)

        decision = self.triage.evaluate(
            frame, tracks=tracks, frame_idx=idx, timestamp=ts
        )
        cand_summary = "; ".join(t.kind.value for t in decision.triggers) or "periyodik_gozetim"

        result = FrameResult(
            frame_idx=idx,
            timestamp_s=video_time,
            tracks=tracks,
            triage=decision,
            high_res_tracker=True,
        )

        image_336, cropped, prompt_suffix = self._build_vlm_input(
            frame, decision, tracks
        )
        trigger_info = f"{cand_summary}{prompt_suffix}"
        result.cropped = cropped
        result.vlm_input_shape = tuple(image_336.shape)
        result.gate_low_res_size = (self.vlm_size, self.vlm_size)
        result.detail_high_res_shape = tuple(image_336.shape)

        is_incident = False
        if decision.triggers:
            kinds = {t.kind for t in decision.triggers}
            high_risk_triggers = {
                TriggerKind.DANGEROUS_MOTION,
                TriggerKind.SSIM,
                TriggerKind.COLOR_FIRE
            }
            if kinds.intersection(high_risk_triggers):
                is_incident = True
                
        if getattr(self, "_force_next_incident", False):
            is_incident = True
            self._force_next_incident = False

        mode = "incident" if is_incident else "routine"

        self._publish_snapshot(
            image_336, idx, fps, tracks, trigger_info, cropped, mode=mode
        )

        if not run_async:
            # Senkron: period + force
            should = self._vlm_period_allowed(ts) or force_scan
            if should:
                self._run_vlm_sync(
                    result,
                    image_336,
                    tracks,
                    idx,
                    fps,
                    ts,
                    trigger_info,
                    cropped,
                    mode=mode,
                )
            self.last_frame_result = result
            return result

        # Async UI: her karede snapshot + periyot doluysa VLM one-shot
        self.schedule_vlm_if_due()
        if self._vlm_busy:
            result.vlm_called = True
            result.gate_called = True
        if self._last_vlm_called_on_frame == idx or idx < self._vlm_triggered_until_frame:
            result.vlm_called = True
            result.gate_called = True
        if self._vlm_call_count > 0:
            result.vlm_called = True
            result.gate_called = True
        result.cropped = self._last_cropped or cropped
        self.last_frame_result = result
        return result

    def _save_report(self, analysis, frame_result, fps=30.0) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        path = self.report_dir / f"sentinel_report_f{frame_result.frame_idx}_{ts}.json"
        payload = analysis.model_dump()
        payload["pipeline_meta"] = {
            "frame_idx": frame_result.frame_idx,
            "architecture": "single_vlm_336_period_2s",
            "vlm_size": self.vlm_size,
            "vlm_period_s": self.vlm_period_s,
            "cropped": getattr(frame_result, "cropped", False),
            "vlm_input_shape": frame_result.vlm_input_shape
            or frame_result.detail_high_res_shape,
            "high_res_tracker": True,
            "high_res_vlm": False,
            "track_ids": [t.track_id for t in frame_result.tracks],
            "triage": frame_result.triage.to_dict() if frame_result.triage else None,
            "tools_executed": [r.to_dict() for r in frame_result.tool_results],
            "fps": fps,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def process_video(
        self,
        source: Union[str, Path, int],
        fps_override: Optional[float] = None,
        max_frames: Optional[int] = None,
        on_frame: Optional[Callable[[FrameResult, np.ndarray], None]] = None,
    ) -> PipelineResult:
        if self.tracker is None or self.agent is None:
            self.ensure_components()
        self.reset()
        limit = max_frames if max_frames is not None else self.max_frames
        src_str = str(source)
        cap = cv2.VideoCapture(source if not isinstance(source, Path) else str(source))
        if not cap.isOpened():
            return PipelineResult(source=src_str, success=False, error=f"Video açılamadı: {src_str}")
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        fps = float(fps_override) if fps_override else (video_fps if video_fps > 1e-3 else 30.0)
        out = PipelineResult(source=src_str, fps=fps)
        t0 = time.perf_counter()
        idx = 0
        try:
            while True:
                if limit is not None and idx >= limit:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                # Senkron modda duvar saati kullan — video zamanı VLM periyodunu bozar
                fr = self.process_frame(frame, frame_idx=idx, fps=fps)
                out.frames_processed += 1
                if fr.vlm_called:
                    out.vlm_calls += 1
                    out.gate_calls += 1
                    if fr.analysis:
                        out.last_analysis = fr.analysis
                        out.last_summary = fr.analysis.summary
                    if fr.report_path:
                        out.reports.append(fr.report_path)
                if self.store_frame_results:
                    out.frame_results.append(fr)
                if on_frame:
                    on_frame(fr, frame)
                idx += 1
        except Exception as exc:
            logger.exception("pipeline")
            out.success = False
            out.error = str(exc)
        finally:
            cap.release()
        out.duration_s = time.perf_counter() - t0
        out.stats = self._collect_stats(out)
        self.last_pipeline_result = out
        return out

    def process_frames(
        self,
        frames: Sequence[np.ndarray],
        fps: float = 30.0,
        source: str = "frame_sequence",
        timestamps: Optional[Sequence[float]] = None,
        yolo_detections_per_frame: Optional[Sequence[Optional[np.ndarray]]] = None,
        on_frame: Optional[Callable[[FrameResult, np.ndarray], None]] = None,
    ) -> PipelineResult:
        if self.tracker is None or self.agent is None:
            self.ensure_components()
        self.reset()
        out = PipelineResult(source=source, fps=fps)
        t0 = time.perf_counter()
        try:
            for i, frame in enumerate(frames):
                if self.max_frames is not None and i >= self.max_frames:
                    break
                ts = (
                    float(timestamps[i])
                    if timestamps is not None and i < len(timestamps)
                    else i / max(fps, 1e-6)
                )
                yolo_dets = None
                if yolo_detections_per_frame is not None and i < len(yolo_detections_per_frame):
                    yolo_dets = yolo_detections_per_frame[i]
                fr = self.process_frame(
                    frame, frame_idx=i, fps=fps, timestamp_s=ts, yolo_detections=yolo_dets
                )
                out.frames_processed += 1
                if fr.vlm_called:
                    out.vlm_calls += 1
                    out.gate_calls += 1
                    if fr.analysis:
                        out.last_analysis = fr.analysis
                        out.last_summary = fr.analysis.summary
                    if fr.report_path:
                        out.reports.append(fr.report_path)
                if self.store_frame_results:
                    out.frame_results.append(fr)
                if on_frame:
                    on_frame(fr, frame)
        except Exception as exc:
            logger.exception("pipeline frames")
            out.success = False
            out.error = str(exc)
        out.duration_s = time.perf_counter() - t0
        out.stats = self._collect_stats(out)
        self.last_pipeline_result = out
        return out

    def iter_video(
        self,
        source: Union[str, Path, int],
        fps_override: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> Generator[Tuple[FrameResult, np.ndarray], None, None]:
        if self.tracker is None or self.agent is None:
            self.ensure_components()
        self.reset()
        limit = max_frames if max_frames is not None else self.max_frames
        cap = cv2.VideoCapture(source if not isinstance(source, Path) else str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Video açılamadı: {source}")
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        fps = float(fps_override) if fps_override else (video_fps if video_fps > 1e-3 else 30.0)
        idx = 0
        try:
            while True:
                if limit is not None and idx >= limit:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                # Senkron modda duvar saati kullan
                yield self.process_frame(frame, frame_idx=idx, fps=fps), frame
                idx += 1
        finally:
            cap.release()

    def _collect_stats(self, out: PipelineResult) -> Dict[str, Any]:
        return {
            "triage": dict(self.triage.stats),
            "architecture": "single_vlm_336",
            "vlm_period_s": self.vlm_period_s,
            "vlm_size": self.vlm_size,
            "vlm_calls": out.vlm_calls,
            "gate_hz": self.gate_hz,
            "gate_size": self.gate_size,
            "gate_calls": out.gate_calls,
            "detail_vlm_calls": out.vlm_calls,
            "vlm_call_ratio": out.vlm_calls / max(out.frames_processed, 1),
            "vlm_savings_approx": max(0.0, 1.0 - out.vlm_calls / max(out.frames_processed, 1)),
            "tool_calls": len(self.tools.call_history),
            "memory": self.agent.memory.as_dict() if self.agent else {},
        }

    def get_last_summary(self) -> str:
        if self._last_vlm_summary and "Henüz VLM çalışmadı" not in self._last_vlm_summary:
            return self._last_vlm_summary
        if self.last_pipeline_result and self.last_pipeline_result.last_summary:
            return self.last_pipeline_result.last_summary
        if self.last_frame_result and self.last_frame_result.analysis:
            return self.last_frame_result.analysis.summary
        return ""

    def get_last_report_json(self) -> Optional[Dict[str, Any]]:
        if self._last_vlm_json and self._last_vlm_json != "{}":
            try:
                return json.loads(self._last_vlm_json)
            except Exception:
                pass
        if self.last_pipeline_result and self.last_pipeline_result.last_analysis:
            return self.last_pipeline_result.last_analysis.model_dump()
        if self.last_frame_result and self.last_frame_result.analysis:
            return self.last_frame_result.analysis.model_dump()
        return None

    def warmup(self) -> None:
        """Eager load + CUDA smoke. CUDA kernel mismatch fatal (yukarı fırlatılır)."""
        from src.vlm.cuda_compat import (
            CudaKernelMismatchError,
            ensure_cuda_kernels_or_raise,
            is_cuda_kernel_mismatch,
            raise_if_cuda_kernel_mismatch,
        )

        logger.warning("[WARMUP] Eager loading ve CUDA warmup başlatılıyor...")
        try:
            # Gerçek VLM yolu: model indirmeden önce torch↔GPU smoke
            if self.agent is not None and getattr(self.agent, "backend", "mock") != "mock":
                if getattr(self.agent, "device", "cuda") == "cuda":
                    ensure_cuda_kernels_or_raise()

            self.ensure_components()
            if self.tracker is not None:
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                self.tracker.process_frame(dummy, run_mog2=True)
                logger.warning("[WARMUP] YOLO ve MOG2 tracker ısındı.")
            if self.agent is not None and getattr(self.agent, "backend", "mock") != "mock":
                if not self.agent.is_loaded:
                    self.agent.load()
                dummy = np.zeros((self.vlm_size, self.vlm_size, 3), dtype=np.uint8)
                try:
                    self.agent.analyze_detail(
                        frame=dummy,
                        frame_idx=0,
                        fps=30.0,
                        tracks=[],
                        trigger_info="warmup",
                        execute_tools=False,
                    )
                except CudaKernelMismatchError:
                    raise
                except Exception as e:
                    raise_if_cuda_kernel_mismatch(e)
                    if is_cuda_kernel_mismatch(e):
                        raise
                    logger.warning("[WARMUP] VLM warmup uyarısı: %s", e)
                logger.warning("[WARMUP] VLM 336 warmup tamam.")
        except CudaKernelMismatchError:
            raise
        except Exception as e:
            raise_if_cuda_kernel_mismatch(e)
            logger.error("[WARMUP] Hata: %s", e)


def build_demo_pipeline(
    report_dir: Optional[Path] = None,
    roi_polygon: Optional[Sequence[Tuple[float, float]]] = None,
    mock_vlm: bool = True,
    gate_size: int = DEFAULT_VLM_SIZE,
    gate_hz: float = DEFAULT_GATE_HZ,
    force_gate_every_candidate: bool = False,
    vlm_backend: str = "mock",
    yolo_device: str = "cpu",
    vlm_size: Optional[int] = None,
    vlm_period_s: Optional[float] = None,
    use_vllm: bool = False,
    **kwargs: Any,
) -> SentinelPipeline:
    if "low_res_size" in kwargs:
        gate_size = kwargs.pop("low_res_size")
    if "scan_hz" in kwargs:
        gate_hz = kwargs.pop("scan_hz")
    if "force_scan_every_frame" in kwargs:
        force_gate_every_candidate = kwargs.pop("force_scan_every_frame")
    if vlm_size is not None:
        gate_size = vlm_size
    if vlm_period_s is not None and vlm_period_s > 0:
        gate_hz = 1.0 / float(vlm_period_s)

    tools = ToolRegistry(report_dir=report_dir or DEFAULT_REPORT_DIR)
    tracker = HybridTracker(yolo_model=_EmptyYolo(), use_ultralytics_botsort=False)
    if kwargs.pop("use_real_yolo", False):
        tracker = HybridTracker(
            model_path=kwargs.pop("yolo_weights", "yolov8n.pt"),
            device=yolo_device,
            use_ultralytics_botsort=False,
        )

    triage = TriageEngine(
        roi_polygon=list(roi_polygon) if roi_polygon else None,
        periodic_interval_s=7.0,
        coalesce_window_ms=900.0,
    )

    if mock_vlm or vlm_backend == "mock":
        agent = InternVLAgent(
            tools=tools,
            generator_fn=make_mock_generator() if mock_vlm else None,
            auto_execute_tools=True,
            use_vllm=use_vllm,
        )
        agent.load()
    else:
        from src.vlm.factory import create_vlm_agent

        agent = create_vlm_agent(
            backend=vlm_backend,  # type: ignore
            tools=tools,
            device=kwargs.pop("vlm_device", "cuda"),
            load_in_8bit=kwargs.pop("load_in_8bit", False),
            load_in_4bit=kwargs.pop("load_in_4bit", False),
            auto_load=True,
            use_vllm=use_vllm,
        )


    return SentinelPipeline(
        tracker=tracker,
        triage=triage,
        agent=agent,
        tools=tools,
        report_dir=report_dir,
        save_reports=True,
        gate_size=gate_size,
        gate_hz=gate_hz,
        force_gate_every_candidate=force_gate_every_candidate,
    )


class _EmptyYolo:
    names = {0: "person"}

    def predict(self, source=None, **kwargs):
        del source, kwargs

        class _Boxes:
            def __len__(self):
                return 0

            @property
            def xyxy(self):
                return self

            @property
            def conf(self):
                return self

            @property
            def cls(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.zeros((0, 4))

        class _R:
            boxes = _Boxes()

        return [_R()]
