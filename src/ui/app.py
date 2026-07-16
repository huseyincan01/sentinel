"""
Sentinel Gradio arayüzü.

- Açılışta model yükleme çubuğu (hazır olana kadar Analiz kapalı)
- Tek VLM @336, ~2 sn; video boyunca sürekli worker (yeni kare snapshot)
- YOLO/MOG2 high-res track + tetikte crop

Çalıştırma:
  uv run python -m src.ui.app
"""

from __future__ import annotations

import html
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline.main_pipeline import (  # noqa: E402
    FrameResult,
    SentinelPipeline,
    build_demo_pipeline,
)
from src.vlm.cuda_compat import (  # noqa: E402
    CudaKernelMismatchError,
    humanize_cuda_error,
    is_cuda_kernel_mismatch,
)

logger = logging.getLogger("sentinel.ui")

PROJECT_ROOT = _ROOT
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
PREVIEW_DIR = PROJECT_ROOT / "outputs" / "previews"
DEFAULT_ROI_FRAC = None  # type: ignore
_USE_VLLM = False


# ---------------------------------------------------------------------------
# Global model durumu (tek yükleme, UI çubuğu)
# ---------------------------------------------------------------------------
_MODEL_LOCK = threading.Lock()
_MODEL: Dict[str, Any] = {
    "status": "idle",  # idle | loading | ready | error
    "progress": 0.0,  # 0..1
    "message": "Model henüz yüklenmedi. Ayarları seçip bekleyin veya yükleme başlar.",
    "error": "",
    "mock": None,
    "backend": None,
    "pipe": None,
}


def _set_model(
    status: Optional[str] = None,
    progress: Optional[float] = None,
    message: Optional[str] = None,
    error: Optional[str] = None,
    pipe: Any = None,
    mock: Any = None,
    backend: Any = None,
) -> None:
    with _MODEL_LOCK:
        if status is not None:
            _MODEL["status"] = status
        if progress is not None:
            _MODEL["progress"] = float(max(0.0, min(1.0, progress)))
        if message is not None:
            _MODEL["message"] = message
        if error is not None:
            _MODEL["error"] = error
        if pipe is not None:
            _MODEL["pipe"] = pipe
        if mock is not None:
            _MODEL["mock"] = mock
        if backend is not None:
            _MODEL["backend"] = backend


def get_model_state() -> Dict[str, Any]:
    with _MODEL_LOCK:
        return dict(_MODEL)


def model_status_html() -> str:
    st = get_model_state()
    status = st["status"]
    pct = int(st["progress"] * 100)
    msg = st["message"] or ""
    err = st["error"] or ""

    if status == "ready":
        color, border, title = "#10B981", "#10B981", f"✅ MODEL HAZIR ({pct}%)"
    elif status == "loading":
        color, border, title = "#F59E0B", "#F59E0B", f"⏳ MODEL YÜKLENİYOR… {pct}%"
    elif status == "error":
        color, border, title = "#EF4444", "#EF4444", "❌ MODEL HATASI"
    else:
        color, border, title = "#9CA3AF", "#374151", "⚪ MODEL BEKLENİYOR"

    err_html = ""
    if err:
        safe = html.escape(err).replace("\n", "<br>")
        err_html = (
            f"<div style='color:#FCA5A5;font-size:0.82em;margin-top:6px;"
            f"font-weight:normal;line-height:1.35;white-space:normal;'>{safe}</div>"
        )
    return (
        f"<div style='background:#1F2937;color:white;padding:14px;border-radius:8px;"
        f"border:2px solid {border};font-weight:bold;'>"
        f"<div style='color:{color};font-size:1.05em;'>{title}</div>"
        f"<div style='color:#D1D5DB;font-size:0.9em;margin-top:6px;font-weight:normal;'>"
        f"{html.escape(msg)}</div>"
        f"{err_html}</div>"
    )


def load_global_model(mock_vlm: bool = False, vlm_backend: str = "smolvlm") -> str:
    """
    Model + YOLO yükle (senkron, arka plan thread'inden çağrılır).
    Dönen: durum HTML.
    """
    st = get_model_state()
    if (
        st["status"] == "ready"
        and st["pipe"] is not None
        and st["mock"] == mock_vlm
        and st["backend"] == (vlm_backend if not mock_vlm else "mock")
    ):
        _set_model(message="Model zaten hazır — analize başlayabilirsiniz.")
        return model_status_html()

    # Yeniden yükleme
    old = st.get("pipe")
    if old is not None:
        try:
            old.stop_vlm_worker()
        except Exception:
            pass

    _set_model(
        status="loading",
        progress=0.05,
        message="Pipeline oluşturuluyor…",
        error="",
        mock=mock_vlm,
        backend=vlm_backend if not mock_vlm else "mock",
        pipe=None,
    )

    stop_vlm_progress = True
    stop_warmup_progress = True
    try:
        _set_model(progress=0.15, message="YOLO + triage kuruluyor…")
        pipe = build_demo_pipeline(
            report_dir=REPORT_DIR,
            roi_polygon=None,
            mock_vlm=mock_vlm,
            vlm_backend=vlm_backend if not mock_vlm else "mock",
            vlm_size=336,
            vlm_period_s=2.0,
            yolo_device="cuda",  # YOLO'yu CPU'dan GPU'ya aldık, canlı izleme kasmasını çözecek
            use_real_yolo=True,
            yolo_weights="yolov8n.pt",
            load_in_4bit=False,  
            use_vllm=_USE_VLLM,
        )
        pipe.triage.periodic_interval_s = 7.0

        _set_model(progress=0.40, message="VLM ağırlıkları belleğe alınıyor (gerçek zamanlı)...")
        if pipe.agent is not None and not pipe.agent.is_loaded:
            pipe.agent.load()

        _set_model(progress=0.80, message="CUDA ilk çıkarım için ısınıyor (warmup)...")
        try:
            pipe.warmup()
        except CudaKernelMismatchError as e:
            raise
        except Exception as e:
            if is_cuda_kernel_mismatch(e):
                raise CudaKernelMismatchError(humanize_cuda_error(e), original=e) from e
            logger.warning("Warmup uyarısı: %s", e)

        _set_model(
            status="ready",
            progress=1.0,
            message=(
                f"Hazır — backend={'mock' if mock_vlm else vlm_backend}. "
                "Video yükleyip Analizi Başlat’a basabilirsiniz."
            ),
            error="",
            pipe=pipe,
            mock=mock_vlm,
            backend=vlm_backend if not mock_vlm else "mock",
        )
        logger.info("Global model HAZIR mock=%s backend=%s", mock_vlm, vlm_backend)
    except Exception as e:
        logger.exception("Model yükleme hatası")
        friendly = humanize_cuda_error(e)
        title = (
            "CUDA mimari uyumsuzluğu (torch ↔ GPU)"
            if is_cuda_kernel_mismatch(e)
            else "Model yüklenemedi"
        )
        _set_model(
            status="error",
            progress=0.0,
            message=title,
            error=friendly,
            pipe=None,
        )

    return model_status_html()



def create_pipeline(
    mock_vlm: bool = True,
    max_frames: Optional[int] = None,
    use_periodic: bool = True,
    vlm_backend: str = "smolvlm",
) -> SentinelPipeline:
    """Geriye uyum / test: global yoksa yeni kur."""
    st = get_model_state()
    if (
        st["status"] == "ready"
        and st["pipe"] is not None
        and st["mock"] == mock_vlm
        and st["backend"] == (vlm_backend if not mock_vlm else "mock")
    ):
        pipe = st["pipe"]
        pipe.max_frames = max_frames
        return pipe

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pipe = build_demo_pipeline(
        report_dir=REPORT_DIR,
        roi_polygon=None,
        mock_vlm=mock_vlm,
        vlm_backend=vlm_backend if not mock_vlm else "mock",
        vlm_size=336,
        vlm_period_s=2.0,
        yolo_device="cuda",
        use_real_yolo=True,
        yolo_weights="yolov8n.pt",
        load_in_4bit=False,
        use_vllm=_USE_VLLM,
    )
    if use_periodic:
        pipe.triage.periodic_interval_s = 7.0
    else:
        pipe.triage.periodic_interval_s = 1e9
    pipe.max_frames = max_frames
    return pipe


def _draw_overlay(
    frame: np.ndarray,
    fr: FrameResult,
    roi=None,
    vlm_busy: bool = False,
    vlm_count: int = 0,
    cropped: bool = False,
    last_risk: str = "",
) -> np.ndarray:
    out = frame.copy()
    if roi and len(roi) >= 3:
        pts = np.array(roi, dtype=np.int32).reshape((-1, 1, 2))
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], (0, 60, 120))
        cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
        cv2.polylines(out, [pts], True, (0, 140, 255), 2)

    for t in fr.tracks:
        x1, y1, x2, y2 = t.bbox
        color = (80, 220, 80) if t.source == "yolo" else (220, 180, 40)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            t.track_id,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    status = "IZLEME"
    color = (180, 180, 180)
    mode = "track"
    crop_tag = "crop" if (cropped or getattr(fr, "cropped", False)) else "full"
    if vlm_busy:
        status = f"VLM 336 ANALIZ... n={vlm_count}"
        color = (0, 200, 255)
        mode = f"vlm-{crop_tag}"
    elif fr.vlm_called or last_risk or vlm_count > 0:
        risk = last_risk
        if fr.analysis is not None:
            risk = fr.analysis.risk or risk
        status = f"VLM 336 | {risk or 'ok'} n={vlm_count}"
        color = (0, 80, 255)
        mode = f"vlm-{crop_tag}"
    elif fr.triage and fr.triage.triggers:
        kind = fr.triage.primary_trigger.kind.value if fr.triage.primary_trigger else "?"
        status = f"ADAY ({kind})"
        color = (0, 220, 180)
        mode = "aday"

    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(
        out,
        f"f={fr.frame_idx}  {status}  [{mode}]",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    return out


def analyze_video(
    video_path: Optional[str],
    max_frames: int = 120,
    mock_vlm: bool = True,
    vlm_backend: str = "smolvlm",
    progress: Any = None,
):
    """Gradio generator: canlı kare + VLM özeti."""

    # Tetikleyici sinyal türü → etiket
    _TRIGGER_LABEL: dict = {
        "ssim":              "SSIM Şoku",
        "stillness":         "Hareketsizlik",
        "dangerous_motion":  "Tehlikeli Hareket",
        "entrance":          "Yeni Giriş",
        "mog2_motion":       "MOG2 Hareket",
        "roi":               "Bölge Tetik",
        "periodic":          "Periyodik",
        "color_fire":        "Yangın/Duman",
    }

    def _build_vlm_html(
        status: str,
        risk: str = "",
        vlm_busy: bool = False,
        vlm_count: int = 0,
        cropped: bool = False,
        error: str = "",
        primary_kind: str = "",
        all_kinds: list | None = None,
    ) -> str:
        base = "background-color:#1F2937;color:white;padding:12px;border-radius:8px;text-align:center;font-size:1.05em;font-weight:bold;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);"
        header = (
            "<span style='color:#9CA3AF;font-size:0.8em;display:block;margin-bottom:6px;'>"
            "TEK VLM · 336×336 · sürekli (~2 sn min aralık)</span>"
        )
        if error or status == "error":
            line = f"<span style='color:#F87171;'>❌ HATA n={vlm_count}</span>"
            border = "border:2px solid #EF4444;"
        elif vlm_busy or status == "analyzing":
            line = f"<span style='color:#F59E0B;'>⏳ ANALİZ… n={vlm_count}</span>"
            border = "border:2px solid #F59E0B;"
        elif status == "danger_found":
            risk_label = risk.upper() if risk else "?"
            line = f"<span style='color:#EF4444;'>🚨 TEHLİKE {risk_label} · n={vlm_count}</span>"
            border = "border:2px solid #EF4444;"
        elif status == "no_danger":
            line = f"<span style='color:#10B981;'>✅ Tehlike yok · n={vlm_count}</span>"
            border = "border:2px solid #10B981;"
        else:
            line = f"<span style='color:#9CA3AF;'>Beklemede · n={vlm_count}</span>"
            border = "border:1px solid #374151;"

        # Girdi odak türü (ROI kaldırıldı, yerine kırpılmış odak/tam kare)
        focus_txt = "Kırpılmış Odak (Crop+20%)" if cropped else "Tam Kare"
        focus_sub = f"<span style='color:#7DD3FC;display:block;margin-top:6px;font-size:0.85em;'>Girdi: {focus_txt}</span>"

        # Tetikleyen sinyal
        if primary_kind:
            trig_name = _TRIGGER_LABEL.get(primary_kind, primary_kind.upper())
            trig_sub = f"<span style='color:#FBBF24;display:block;margin-top:4px;font-size:0.82em;'>Tetikleyen: {trig_name}</span>"
        else:
            trig_sub = "<span style='color:#6B7280;display:block;margin-top:4px;font-size:0.82em;'>Tetikleyen: Beklemede</span>"

        err_sub = ""
        if error:
            # CUDA rehber mesajı uzun; kısaltma eşiğini yükselt
            friendly = humanize_cuda_error(error)
            limit = 520 if is_cuda_kernel_mismatch(friendly) else 160
            shown = friendly if len(friendly) <= limit else friendly[:limit] + "…"
            safe = html.escape(shown).replace("\n", "<br>")
            err_sub = (
                f"<span style='color:#FCA5A5;display:block;margin-top:4px;"
                f"font-size:0.72em;line-height:1.3;text-align:left;font-weight:normal;'>"
                f"{safe}</span>"
            )
        return f"<div style='{base}{border}'>{header}{line}{focus_sub}{trig_sub}{err_sub}</div>"

    default_json = '{\n  "durum": "Tehlike tespit edilmedi / Beklemede"\n}'



    if not video_path:
        yield None, None, "Lütfen video yükleyin.", default_json, "⚠️ Video yükleyin.", _build_vlm_html("idle")
        return

    path = Path(video_path)
    if not path.is_file():
        yield None, None, "", default_json, f"❌ Video yok: {video_path}", _build_vlm_html("idle")
        return

    # Model hazır mı?
    st = get_model_state()
    want_backend = "mock" if mock_vlm else (vlm_backend or "smolvlm")
    if st["status"] != "ready" or st["pipe"] is None or st["backend"] != want_backend:
        if mock_vlm:
            # Mock modu: anında pipeline kur (hızlı, test için)
            try:
                pipe = create_pipeline(mock_vlm=True)
            except Exception as e:
                yield (
                    None,
                    None,
                    f"Pipeline oluşturulamadı: {e}",
                    default_json,
                    f"❌ Mock pipeline hatası: {e}",
                    _build_vlm_html("error", error=str(e)),
                )
                return
        else:
            # Gerçek model: ağırlıklar diskte, RAM/VRAM'e al (blocking — kasıtlı).
            # Önce kullanıcıya durum bildir, ardından yükle.
            yield (
                None,
                None,
                "Model ağırlıkları yükleniyor…",
                default_json,
                "⏳ Model ağırlıkları RAM/VRAM'e alınıyor, lütfen bekleyin…",
                _build_vlm_html("idle"),
            )
            load_global_model(mock_vlm=mock_vlm, vlm_backend=vlm_backend or "smolvlm")
            st = get_model_state()
            if st["status"] != "ready" or st["pipe"] is None:
                yield (
                    None,
                    None,
                    f"Model yüklenemedi: {st.get('error')}",
                    default_json,
                    f"❌ {st.get('error') or st.get('message')}",
                    _build_vlm_html("error", error=st.get("error") or ""),
                )
                return
            pipe = st["pipe"]

    else:
        pipe: SentinelPipeline = st["pipe"]

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        yield None, None, "", default_json, f"❌ Video açılamadı: {path.name}", _build_vlm_html("idle")
        return



    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    if fps < 1e-3:
        fps = 15.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    limit = int(max_frames) if max_frames and max_frames > 0 else None

    preview_path = PREVIEW_DIR / f"preview_{path.stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(preview_path), fourcc, fps, (w, h))

    frames_done = 0
    last_summary = "İzleme aktif — VLM sürekli çalışacak."
    last_json = default_json
    vlm_calls = 0
    vis_rgb = None

    try:
        # Sayaçları sıfırla ama modeli BOŞALTMA
        pipe.stop_vlm_worker()
        pipe._frame_idx = 0
        pipe._frame_buffer.clear()
        pipe._vlm_call_count = 0
        pipe._gate_call_count = 0
        pipe._detail_call_count = 0
        pipe._vlm_status = "idle"
        pipe._vlm_last_risk = ""
        pipe._last_vlm_summary = "Henüz VLM çalışmadı."
        pipe._last_vlm_json = "{}"
        pipe._last_vlm_called_on_frame = -1
        pipe._last_cropped = False
        pipe._last_vlm_error = ""
        pipe._next_vlm_not_before = None
        pipe._vlm_busy = False
        with pipe._snap_lock:
            pipe._snapshot = None
        if pipe.tracker is not None and hasattr(pipe.tracker, "reset"):
            pipe.tracker.reset()
        if pipe.triage is not None:
            pipe.triage.reset()
        if pipe.agent is not None:
            pipe.agent.reset_memory()
            if pipe.agent.tools is not None:
                pipe.agent.tools.reset_history()
        pipe.tools.reset_history()

        # Sürekli VLM worker — video boyunca kesilmeden
        pipe.start_vlm_worker()

        idx = 0
        while True:
            if limit is not None and idx >= limit:
                break

            t_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            fr = pipe.process_frame(
                frame,
                frame_idx=idx,
                fps=float(fps),
                timestamp_s=idx / float(fps),
                run_async=True,
            )
            vlm_calls = int(getattr(pipe, "_vlm_call_count", 0))
            vlm_busy = bool(getattr(pipe, "_vlm_busy", False))
            cropped = bool(
                getattr(pipe, "_last_cropped", False) or getattr(fr, "cropped", False)
            )
            last_risk = getattr(pipe, "_vlm_last_risk", "") or ""

            vis = _draw_overlay(
                frame,
                fr,
                roi=None,
                vlm_busy=vlm_busy,
                vlm_count=vlm_calls,
                cropped=cropped,
                last_risk=last_risk,
            )
            if writer.isOpened():
                if vis.shape[1] != w or vis.shape[0] != h:
                    vis = cv2.resize(vis, (w, h))
                writer.write(vis)

            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

            if pipe.get_last_summary():
                last_summary = pipe.get_last_summary()
            rep = pipe.get_last_report_json()
            if rep:
                last_json = json.dumps(rep, ensure_ascii=False, indent=2)

            vlm_status = getattr(pipe, "_vlm_status", "idle")
            if vlm_busy and vlm_status == "idle":
                vlm_status = "analyzing"
            vlm_err = getattr(pipe, "_last_vlm_error", "") or ""

            # Tetikleyici sinyal bilgisi
            if fr.triage and fr.triage.triggers:
                all_kinds = [t.kind.value for t in fr.triage.triggers]
                primary_kind = (
                    fr.triage.primary_trigger.kind.value
                    if fr.triage.primary_trigger
                    else all_kinds[0]
                )
            else:
                all_kinds = []
                primary_kind = ""

            current_html = _build_vlm_html(
                vlm_status,
                last_risk,
                vlm_busy=vlm_busy,
                vlm_count=vlm_calls,
                cropped=cropped,
                error=vlm_err,
                primary_kind=primary_kind,
                all_kinds=all_kinds,
            )

            frames_done += 1
            idx += 1

            busy_note = " | VLM⚡" if vlm_busy else ""
            crop_note = " | crop" if cropped else ""
            err_note = f" | ERR: {vlm_err[:50]}" if vlm_err else ""
            # VLM'nin işlediği kareyi al ve çerçeve çiz
            vlm_image = None
            with pipe._snap_lock:
                if pipe._snapshot is not None:
                    vlm_bgr = pipe._snapshot["image"]
                    vlm_rgb = cv2.cvtColor(vlm_bgr, cv2.COLOR_BGR2RGB)
                    
                    # Risk durumuna göre ince renkli çerçeve çiz
                    h_v, w_v = vlm_rgb.shape[:2]
                    vlm_status = getattr(pipe, "_vlm_status", "idle")
                    if vlm_status == "danger_found":
                        color = (239, 68, 68)  # İnce Kırmızı
                    elif vlm_status == "no_danger":
                        color = (16, 185, 129) # İnce Yeşil
                    elif vlm_status == "analyzing":
                        color = (245, 158, 11)  # İnce Turuncu
                    else:
                        color = (156, 163, 175) # İnce Gri
                    
                    cv2.rectangle(vlm_rgb, (0, 0), (w_v - 1, h_v - 1), color, 3)
                    vlm_image = vlm_rgb

            yield (
                vis_rgb,
                vlm_image,
                last_summary,
                last_json,
                f"İşleniyor: Kare {frames_done}/{limit or total} | VLM@336: {vlm_calls}{busy_note}{crop_note}{err_note}",
                current_html,
            )



            # Gerçek zaman hissi (video FPS)
            elapsed = time.perf_counter() - t_start
            delay = (1.0 / float(fps)) - elapsed
            if delay > 0:
                time.sleep(delay)

            if progress is not None:
                denom = limit or (total if total > 0 else max(frames_done, 1))
                try:
                    progress(
                        min(1.0, frames_done / max(denom, 1)),
                        desc=f"Kare {frames_done} | VLM {vlm_calls}",
                    )
                except Exception:
                    pass
    finally:
        # Son VLM bitsin; worker'ı kapat (model global kalır)
        try:
            pipe.wait_vlm_idle(timeout_s=180.0)
            if int(getattr(pipe, "_vlm_call_count", 0)) == 0 and frames_done > 0:
                pipe.ensure_at_least_one_vlm()
                pipe.wait_vlm_idle(timeout_s=180.0)
        except Exception:
            pass
        try:
            pipe.stop_vlm_worker()
        except Exception:
            pass
        cap.release()
        writer.release()

    vlm_calls = int(getattr(pipe, "_vlm_call_count", vlm_calls))
    savings = (
        max(0.0, 1.0 - (vlm_calls / frames_done)) * 100.0 if frames_done > 0 else 0.0
    )
    if pipe.get_last_summary():
        last_summary = pipe.get_last_summary()
    rep = pipe.get_last_report_json()
    if rep:
        last_json = json.dumps(rep, ensure_ascii=False, indent=2)

    err = getattr(pipe, "_last_vlm_error", "") or ""
    err_line = f"\nVLM hata: {err}" if err else ""
    if vlm_calls == 0:
        status = (
            f"⚠️ VLM 0 çağrı — {path.name}\n"
            f"Kare: {frames_done}{err_line}\n"
            f"Model durum çubuğunu kontrol edin."
        )
    else:
        status = (
            f"✅ Tamamlandı — {path.name}\n"
            f"Kare: {frames_done} | VLM@336: {vlm_calls} | tasarruf ~%{savings:.0f}\n"
            f"Sürekli worker: her bitişte yeni kare (min ~2 sn aralık)\n"
            f"Raporlar: {REPORT_DIR} | Önizleme: {preview_path.name}\n"
            f"Mod: {'Mock' if mock_vlm else vlm_backend}{err_line}"
        )

    vlm_image = None
    with pipe._snap_lock:
        if pipe._snapshot is not None:
            vlm_bgr = pipe._snapshot["image"]
            vlm_rgb = cv2.cvtColor(vlm_bgr, cv2.COLOR_BGR2RGB)
            h_v, w_v = vlm_rgb.shape[:2]
            vlm_status = getattr(pipe, "_vlm_status", "idle")
            if vlm_status == "danger_found":
                color = (239, 68, 68)
            elif vlm_status == "no_danger":
                color = (16, 185, 129)
            elif vlm_status == "analyzing":
                color = (245, 158, 11)
            else:
                color = (156, 163, 175)
            cv2.rectangle(vlm_rgb, (0, 0), (w_v - 1, h_v - 1), color, 3)
            vlm_image = vlm_rgb

    yield (
        vis_rgb,
        vlm_image,
        last_summary,
        last_json,
        status,
        _build_vlm_html(
            getattr(pipe, "_vlm_status", "idle"),
            getattr(pipe, "_vlm_last_risk", ""),
            vlm_busy=False,
            vlm_count=vlm_calls,
            cropped=bool(getattr(pipe, "_last_cropped", False)),
            error=err,
            primary_kind="",
        ),
    )




UI_CSS = """
.gradio-container { max-width: 1100px !important; }
#title { text-align: center; margin-bottom: 0.25rem; }
"""


def build_ui():
    import gradio as gr

    with gr.Blocks(
        title="Sentinel — TEKNOFEST 2026",
        css=UI_CSS,
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# 🛡️ Sentinel\n"
            "**Dinamik Hibrit Görsel Algı ve Karar Destek Ajanı**  \n"
            "TEKNOFEST 2026 · *Önce model yüklensin, sonra analiz*",
            elem_id="title",
        )

        model_html = gr.HTML(value=model_status_html(), label="Model Durumu")

        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.Video(label="📹 Video yükle", sources=["upload"])
                max_frames = gr.State(None)
                mock_vlm = gr.Checkbox(
                    value=False,
                    label="Mock VLM (model indirmeden demo)",
                    info="Kapalı = gerçek smolvlm/internvl2",
                )
                vlm_backend = gr.Dropdown(
                    choices=["smolvlm", "internvl2"],
                    value="smolvlm",
                    label="VLM backend",
                )
                load_btn = gr.Button("⬇ Modelı Yükle / Yenile", variant="secondary")
                run_btn = gr.Button(
                    "▶ Analizi Başlat (model hazır olunca)",
                    variant="primary",
                    interactive=False,
                )
                gr.Markdown(
                    "1) **Modelı Yükle** → çubuk %100  \n"
                    "2) Video yükle → **Analizi Başlat**  \n"
                    "VLM video boyunca **sürekli** yeni karelere bakar (min ~2 sn aralık)."
                )
                trigger_status = gr.HTML(
                    value="<div style='background:#1F2937;color:white;padding:12px;border-radius:8px;"
                    "text-align:center;border:1px solid #374151;font-weight:bold;'>"
                    "<span style='color:#9CA3AF;font-size:0.8em;display:block;margin-bottom:6px;'>"
                    "TEK VLM · 336×336 · sürekli (~2 sn min aralık)</span>"
                    "<span style='color:#9CA3AF;'>Beklemede · n=0</span>"
                    "<span style='color:#7DD3FC;display:block;margin-top:6px;font-size:0.85em;'>Girdi: Tam Kare</span>"
                    "<span style='color:#6B7280;display:block;margin-top:4px;font-size:0.82em;'>Tetikleyen: Beklemede</span></div>"
                )
                status = gr.Textbox(label="Durum", lines=5, interactive=False)


            with gr.Column(scale=1):
                video_out = gr.Image(label="🎬 Canlı önizleme")
                vlm_image_out = gr.Image(label="🧠 Yapay Zekanın İşlediği Kare (336x336)", interactive=False)
                report = gr.Code(label="📋 JSON", language="json", lines=14)

        with gr.Row():
            summary = gr.Textbox(label="🇹🇷 Türkçe Aksiyon Odaklı Rapor (Özet)", lines=5, interactive=False)

        gr.Markdown(
            "### Mimari\n"
            "YOLO+MOG2 (high-res) → snapshot 336 · sürekli VLM worker · tetikte crop+%20"
        )


        def _do_load(mock, backend):
            html = load_global_model(mock_vlm=bool(mock), vlm_backend=backend or "smolvlm")
            st = get_model_state()
            ready = st["status"] == "ready"
            return (
                html,
                gr.update(interactive=ready),
                "Model hazır — Analizi Başlat" if ready else st.get("message", ""),
            )

        _LAST_POLL_HTML = None

        def _poll_status():
            nonlocal _LAST_POLL_HTML
            st = get_model_state()
            html = model_status_html()
            if html == _LAST_POLL_HTML:
                html_update = gr.update()
            else:
                _LAST_POLL_HTML = html
                html_update = html

            ready = st["status"] == "ready"
            return (
                html_update,
                gr.update(interactive=ready),
            )


        load_btn.click(
            fn=_do_load,
            inputs=[mock_vlm, vlm_backend],
            outputs=[model_html, run_btn, status],
        )
        # Backend değişince run'ı kilitle, yeniden yüklemeyi hatırlat
        def _on_backend_change(mock, backend):
            st = get_model_state()
            want = "mock" if mock else (backend or "smolvlm")
            if st["status"] == "ready" and st["backend"] == want:
                return model_status_html(), gr.update(interactive=True)
            # Eski pipe'in worker'ını durdur — akşi n VRAM serbest kalsın
            old_pipe = st.get("pipe")
            if old_pipe is not None:
                try:
                    old_pipe.stop_vlm_worker()
                except Exception:
                    pass
            _set_model(
                status="idle",
                progress=0.0,
                message="Ayar değişti — «Modelı Yükle»'ye basın.",
                pipe=None,
            )
            return model_status_html(), gr.update(interactive=False)

        mock_vlm.change(
            fn=_on_backend_change,
            inputs=[mock_vlm, vlm_backend],
            outputs=[model_html, run_btn],
        )
        vlm_backend.change(
            fn=_on_backend_change,
            inputs=[mock_vlm, vlm_backend],
            outputs=[model_html, run_btn],
        )

        run_btn.click(
            fn=analyze_video,
            inputs=[video_in, max_frames, mock_vlm, vlm_backend],
            outputs=[video_out, vlm_image_out, summary, report, status, trigger_status],
        )



        # Sayfa açılınca otomatik smolvlm yükle (arka plan)
        def _autoload():
            # Non-blocking: thread içinde yükle, UI'yı hemen döndür
            def _bg():
                load_global_model(mock_vlm=False, vlm_backend="smolvlm")

            threading.Thread(target=_bg, daemon=True, name="model-autoload").start()
            return model_status_html(), gr.update(interactive=False)

        demo.load(
            fn=_autoload,
            inputs=None,
            outputs=[model_html, run_btn],
        )

        # Periyodik durum tazeleme (yükleme çubuğu)
        timer = gr.Timer(1.0)
        timer.tick(
            fn=_poll_status,
            inputs=None,
            outputs=[model_html, run_btn],
        )

    return demo


def main(share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7860, use_vllm: bool = False):
    global _USE_VLLM
    _USE_VLLM = use_vllm
    import gradio as gr

    logging.basicConfig(level=logging.INFO)
    demo = build_ui()

    for port in range(server_port, server_port + 10):
        try:
            demo.queue().launch(
                share=share,
                server_name=server_name,
                server_port=port,
                show_error=True,
                inbrowser=True,
            )
            break
        except OSError as e:
            if "port" in str(e).lower() or "address already in use" in str(e).lower():
                print(f"Port {port} kullanımda, {port + 1} deneniyor...")
                continue
            raise e


if __name__ == "__main__":
    main()
