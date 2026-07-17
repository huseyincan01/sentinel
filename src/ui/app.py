"""
Sentinel Gradio arayüzü (Basitleştirilmiş)
"""

from __future__ import annotations
import logging
import sys
import threading
import time
from pathlib import Path
import cv2
import numpy as np
import gradio as gr

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))

from src.pipeline.main_pipeline import build_demo_pipeline

logger = logging.getLogger("sentinel.ui")

_MODEL_LOCK = threading.Lock()
_PIPELINE = None
_LOADING_STATUS = "Bekleniyor..."

def load_pipeline(mock: bool, backend: str):
    global _PIPELINE, _LOADING_STATUS
    with _MODEL_LOCK:
        try:
            _LOADING_STATUS = "Yükleniyor..."
            _PIPELINE = build_demo_pipeline(mock_vlm=mock, vlm_backend=backend, vlm_size=336, vlm_period_s=2.0)
            _LOADING_STATUS = "Hazır"
        except Exception as e:
            _LOADING_STATUS = f"Hata: {e}"
    return _LOADING_STATUS

def process_video_generator(video_path: str):
    global _PIPELINE
    if not _PIPELINE:
        yield None, "Lütfen önce modeli yükleyin.", "{}"
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield None, "Video açılamadı.", "{}"
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    _PIPELINE.start_vlm_worker()
    
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Kırmızı/Mavi renk kanalları (Gradio için RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # İşleme
            fr = _PIPELINE.process_frame(frame_rgb, frame_idx=frame_idx, fps=fps, run_async=True)
            
            # Çizim (Overlay)
            out = frame_rgb.copy()
            for t in fr.tracks:
                x1, y1, x2, y2 = t.bbox
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(out, t.track_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Bilgi yazdırma
            cv2.putText(out, f"VLM: {_PIPELINE._vlm_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(out, f"Risk: {_PIPELINE._vlm_last_risk}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            summary = _PIPELINE.get_last_summary()
            score_text = str(getattr(_PIPELINE, "_vlm_last_score", 0.0))
            
            # 30 FPS hızında oynatma efekti
            time.sleep(1.0 / fps)
            yield out, summary, score_text
            frame_idx += 1
            
    finally:
        cap.release()
        _PIPELINE.stop_vlm_worker()

def build_app():
    with gr.Blocks(title="Sentinel") as demo:
        gr.Markdown("# Sentinel: Dinamik Hibrit Görsel Algı ve Karar Destek Ajanı")
        
        with gr.Row():
            mock_cb = gr.Checkbox(label="Mock VLM Kullan", value=True)
            backend_dd = gr.Dropdown(choices=["smolvlm", "internvl2", "mock"], value="mock", label="Backend")
            load_btn = gr.Button("Modeli Yükle")
            status_txt = gr.Textbox(label="Durum", interactive=False)
            
        with gr.Row():
            video_input = gr.Video(label="Girdi Video")
            start_btn = gr.Button("Analizi Başlat")
            
        with gr.Row():
            img_out = gr.Image(label="Canlı Analiz")
            
        with gr.Row():
            summary_out = gr.Textbox(label="VLM Özeti", lines=3)
            score_out = gr.Textbox(label="Ham Risk Skoru", lines=1)

        load_btn.click(fn=load_pipeline, inputs=[mock_cb, backend_dd], outputs=status_txt)
        start_btn.click(fn=process_video_generator, inputs=[video_input], outputs=[img_out, summary_out, score_out])

    return demo

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = build_app()
    app.launch()
