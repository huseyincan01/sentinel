"""
Sentinel pipeline — Tek Hat VLM Mimarisi (Basitleştirilmiş Sürüm)

1) High-res YOLO/MOG2 ile algı ve kırpma kararı
2) Tek VLM @ 336x336, ~2 sn duvar saati periyodu ile çalışır
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Sequence, Union

import cv2
import numpy as np

from src.decision.triage_engine import TriageDecision, TriageEngine
from src.tracking.hybrid_tracker import FrameTracks, HybridTracker, TrackedObject
from src.vlm.internvl_agent import InternVLAgent
from src.vlm.factory import create_vlm_agent
from src.vlm.schemas import AnalysisResult
from src.vlm.tools import ToolRegistry, ToolResult

logger = logging.getLogger("sentinel.pipeline")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
DEFAULT_VLM_SIZE = 336
DEFAULT_VLM_PERIOD_S = 2.0

@dataclass
class FrameResult:
    frame_idx: int
    timestamp_s: float
    tracks: List[TrackedObject] = field(default_factory=list)
    triage: Optional[TriageDecision] = None
    vlm_called: bool = False
    analysis: Optional[AnalysisResult] = None
    tool_results: List[ToolResult] = field(default_factory=list)
    report_path: Optional[str] = None
    cropped: bool = False
    
    # Geriye uyumluluk için dummy alanlar (eski testlerin çökmemesi için)
    gate_called: bool = False
    vlm_input_shape: Optional[Tuple[int, ...]] = None

@dataclass
class PipelineResult:
    source: str
    frames_processed: int = 0
    vlm_calls: int = 0
    gate_calls: int = 0
    reports: List[str] = field(default_factory=list)
    frame_results: List[FrameResult] = field(default_factory=list)
    last_analysis: Optional[AnalysisResult] = None
    last_summary: str = ""
    fps: float = 30.0
    duration_s: float = 0.0
    stats: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "frames_processed": self.frames_processed,
            "vlm_calls": self.vlm_calls,
            "last_summary": self.last_summary,
            "success": self.success,
            "error": self.error,
        }

def get_motion_crop_box(frame: np.ndarray, tracks: Sequence[TrackedObject], padding_pct: float = 0.2) -> Optional[Tuple[int, int, int, int]]:
    """Basitleştirilmiş bounding box birleşimi + dolgu"""
    if not tracks:
        return None
    h_img, w_img = frame.shape[:2]
    xs1, ys1, xs2, ys2 = [], [], [], []
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        xs1.append(int(x1)); ys1.append(int(y1))
        xs2.append(int(x2)); ys2.append(int(y2))
    x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    px, py = int(w * padding_pct), int(h * padding_pct)
    xa = max(0, x1 - px)
    ya = max(0, y1 - py)
    xb = min(w_img, x2 + px)
    yb = min(h_img, y2 + py)
    return (xa, ya, xb - xa, yb - ya)

def build_demo_pipeline(
    report_dir: Optional[Union[str, Path]] = None,
    roi_polygon: Optional[Sequence[Tuple[float, float]]] = None,
    mock_vlm: bool = True,
    vlm_backend: str = "mock",
    tracker_device: str = "cpu",
    **kwargs
) -> SentinelPipeline:
    """Tek noktadan demo pipeline oluşturma (Geriye uyumlu factory)"""
    tools = ToolRegistry(report_dir=report_dir)
    backend = "mock" if mock_vlm else vlm_backend
    triage = TriageEngine(roi_polygon=roi_polygon)
    
    import torch
    vlm_device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = create_vlm_agent(
        backend=backend,
        tools=tools,
        device=vlm_device,
        auto_load=False,
    )
    # VLM ana GPU bütçesini kullanır. YOLO CPU'da high-res çalışarak aynı T4
    # üzerinde inference yarışını ve ilk karedeki CUDA beklemesini önler.
    tracker = HybridTracker(device=tracker_device, use_ultralytics_botsort=False)
    
    return SentinelPipeline(
        tracker=tracker,
        triage=triage,
        agent=agent,
        tools=tools,
        report_dir=report_dir,
        **kwargs
    )

class SentinelPipeline:
    def __init__(
        self,
        tracker: Optional[HybridTracker] = None,
        triage: Optional[TriageEngine] = None,
        agent: Optional[InternVLAgent] = None,
        tools: Optional[ToolRegistry] = None,
        report_dir: Optional[Union[str, Path]] = None,
        vlm_size: int = DEFAULT_VLM_SIZE,
        vlm_period_s: float = DEFAULT_VLM_PERIOD_S,
        **kwargs
    ):
        self.report_dir = Path(report_dir) if report_dir else DEFAULT_REPORT_DIR
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.vlm_size = int(vlm_size)
        self.vlm_period_s = float(vlm_period_s)
        self.tools = tools or ToolRegistry(report_dir=self.report_dir)
        self.tracker = tracker
        self.triage = triage or TriageEngine()
        self.agent = agent

        # Çalışma değişkenleri
        self._frame_idx = 0
        self._vlm_call_count = 0
        self._last_vlm_summary = "Henüz VLM çalışmadı."
        self._vlm_last_score = 0.0
        self._vlm_last_risk = ""
        self._vlm_status = "idle"
        self._last_vlm_error = ""
        self._last_cropped = False
        
        # Snapshot state
        self._snap_lock = threading.Lock()
        self._snapshot: Optional[Dict[str, Any]] = None
        
        # Thread worker state
        self._vlm_worker_thread: Optional[threading.Thread] = None
        self._vlm_worker_stop = threading.Event()
        self._vlm_busy = False
        self._next_vlm_time = 0.0

    def start_vlm_worker(self) -> None:
        if self._vlm_worker_thread and self._vlm_worker_thread.is_alive():
            return
        if self.agent and not self.agent.is_loaded:
            try:
                self.agent.load()
            except Exception as e:
                logger.error(f"Model yüklenemedi: {e}")
        
        self._vlm_worker_stop.clear()
        self._vlm_worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._vlm_worker_thread.start()

    def prepare_for_streaming(self) -> None:
        """Pahalı lazy-load işlemlerini video akışından önce tamamla.

        Gradio'da ilk `yield`, bu işlemlerden sonra gelir. Bu yüzden VLM ve YOLO
        başlangıcını "Modeli Yükle" adımına taşırız; worker burada başlatılmaz.
        """
        if self.agent and not self.agent.is_loaded:
            self.agent.load()
        if self.tracker is not None:
            # Yalnızca model nesnesini hazırlar; gerçek videodan önce GPU/CPU yükü yoktur.
            _ = self.tracker.yolo

    def stop_vlm_worker(self) -> None:
        self._vlm_worker_stop.set()
        if self._vlm_worker_thread and self._vlm_worker_thread.is_alive():
            self._vlm_worker_thread.join(timeout=3.0)
        self._vlm_worker_thread = None

    def wait_vlm_idle(self, timeout_s: float = 120.0) -> None:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while self._vlm_busy and time.monotonic() < deadline:
            time.sleep(0.1)

    def _worker_loop(self) -> None:
        while not self._vlm_worker_stop.is_set():
            time.sleep(0.05)
            if self._vlm_busy or time.monotonic() < self._next_vlm_time:
                continue

            with self._snap_lock:
                snap = self._snapshot
                self._snapshot = None # Sadece en tazesini 1 kere işle
            
            if not snap or not self.agent:
                continue

            self._vlm_busy = True
            self._vlm_status = "analyzing"
            try:
                if not self.agent.is_loaded and getattr(self.agent, "backend", "mock") != "mock":
                    self.agent.load()
                
                analysis = self.agent.analyze_detail(
                    frame=snap["image"],
                    frame_idx=snap["frame_idx"],
                    fps=snap["fps"],
                    tracks=snap["tracks"],
                    trigger_info=snap["trigger_info"],
                    execute_tools=True,
                )
                self._vlm_call_count += 1
                self._last_vlm_summary = analysis.summary
                self._vlm_last_score = analysis.risk_score
                self._vlm_last_risk = analysis.risk or ""
                self._last_cropped = snap["cropped"]
                
                risk_lower = (analysis.risk or "").lower()
                self._vlm_status = "danger_found" if risk_lower in ("yüksek", "kritik", "high", "critical") else "no_danger"

                if self.report_dir:
                    rpath = self.report_dir / f"vlm_report_{snap['frame_idx']:06d}.txt"
                    with open(rpath, "w", encoding="utf-8") as f:
                        import json
                        json.dump(analysis.model_dump_report(), f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                logger.error(f"VLM Hatası: {e}")
                self._last_vlm_error = str(e)
                self._vlm_status = "error"
            finally:
                self._next_vlm_time = time.monotonic() + self.vlm_period_s
                self._vlm_busy = False

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        fps: float = 30.0,
        timestamp_s: Optional[float] = None,
        run_async: bool = False,
        **kwargs
    ) -> FrameResult:
        """Yeni Tek-Hat işleme: Tracker -> Triage -> Crop -> Snapshot"""
        ts = timestamp_s if timestamp_s is not None else (frame_idx / max(fps, 1e-6))
        
        tracks_obj = None
        if self.tracker:
            tracks_obj = self.tracker.process_frame(frame)
            tracks_list = tracks_obj.tracks
        else:
            tracks_list = []

        decision = self.triage.evaluate(frame, tracks=tracks_list, frame_idx=frame_idx, timestamp=ts)
        
        # Crop belirleme
        cropped = False
        vlm_img = frame
        trigger_info = "Tam Kare"

        if decision.has_motion:
            cbox = get_motion_crop_box(frame, tracks_list)
            if cbox:
                x, y, w, h = cbox
                if w >= 2 and h >= 2:
                    vlm_img = frame[y:y+h, x:x+w].copy()
                    cropped = True
                    trigger_info = "Odak (Crop+20%)"
        
        # 336x336 boyutlandırma
        vlm_img_336 = cv2.resize(vlm_img, (self.vlm_size, self.vlm_size), interpolation=cv2.INTER_AREA)

        # Snapshot yayınlama (Background worker alıp işleyecek)
        if run_async:
            with self._snap_lock:
                self._snapshot = {
                    "image": vlm_img_336.copy(),
                    "frame_idx": frame_idx,
                    "fps": fps,
                    "tracks": tracks_list,
                    "trigger_info": trigger_info,
                    "cropped": cropped
                }
        else:
            # Senkron işleme (Testler için)
            if self.agent and (time.monotonic() >= self._next_vlm_time):
                try:
                    if not self.agent.is_loaded:
                        self.agent.load()
                    analysis = self.agent.analyze_detail(
                        frame=vlm_img_336,
                        frame_idx=frame_idx,
                        fps=fps,
                        tracks=tracks_list,
                        trigger_info=trigger_info,
                        execute_tools=True,
                    )
                    self._vlm_call_count += 1
                    self._last_vlm_summary = analysis.summary
                    self._vlm_last_score = analysis.risk_score
                    self._vlm_last_risk = analysis.risk or ""
                    self._last_cropped = cropped
                    self._next_vlm_time = time.monotonic() + self.vlm_period_s
                except Exception as e:
                    self._last_vlm_error = str(e)

        return FrameResult(
            frame_idx=frame_idx,
            timestamp_s=ts,
            tracks=tracks_list,
            triage=decision,
            cropped=cropped,
            vlm_called=(not run_async and time.monotonic() < self._next_vlm_time),
            vlm_input_shape=(self.vlm_size, self.vlm_size, 3)
        )

    def process_frames(self, frames: List[np.ndarray], fps: float = 30.0, **kwargs) -> PipelineResult:
        """Test uyumluluğu için senkron çerçeve işleyici."""
        results = []
        for i, frame in enumerate(frames):
            res = self.process_frame(frame, frame_idx=i, fps=fps, run_async=False)
            results.append(res)
        
        return PipelineResult(
            source="memory",
            frames_processed=len(frames),
            vlm_calls=self._vlm_call_count,
            gate_calls=self._vlm_call_count,  # Backward compat
            frame_results=results,
            last_summary=self._last_vlm_summary,
            success=True
        )

    def process_video(self, video_path: str, max_frames: Optional[int] = None) -> PipelineResult:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return PipelineResult(source=video_path, success=False, error="Video açılamadı")
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        idx = 0
        while True:
            if max_frames and idx >= max_frames: break
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
            idx += 1
        cap.release()
        return self.process_frames(frames, fps=fps)

    def get_last_summary(self) -> str:
        return self._last_vlm_summary

    def get_last_score(self) -> float:
        return self._vlm_last_score
