"""
Sentinel Benchmark Runner — TEKNOFEST 2026

Bu script, benchmarkdosyalari/ içindeki video ve etiketleri kullanarak
Sentinel pipeline'ının performansını, kaza yakalama başarı oranını (Recall/Precision),
VLM tasarruf oranını ve işlem gecikmesini (latency) otomatik olarak ölçer.


Kullanım:
  uv run python benchmark.py --backend mock --limit 5
  uv run python benchmark.py --backend smolvlm --max-frames 100
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.pipeline.main_pipeline import SentinelPipeline, build_demo_pipeline

# Logging ayarları
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("sentinel.benchmark")

PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmarkdosyalari"
LABELS_CSV = BENCHMARK_DIR / "labels.csv"
VIDEOS_DIR = BENCHMARK_DIR / "videos"
OUTPUT_REPORT_JSON = PROJECT_ROOT / "outputs" / "benchmark_report.json"


def load_labels() -> List[Dict[str, str]]:
    if not LABELS_CSV.is_file():
        raise FileNotFoundError(f"labels.csv bulunamadı: {LABELS_CSV}")
    
    entries = []
    with LABELS_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("video"):
                entries.append({
                    "video": row["video"],
                    "label": row["label"]
                })
    return entries


def run_benchmark(
    backend: str = "mock",
    limit: Optional[int] = None,
    max_frames_per_video: Optional[int] = 120,
    save_reports: bool = False,
    use_vllm: bool = False,
):
    print("=" * 60)
    print(f"SENTINEL BENCHMARK RUNNER | Backend: {backend.upper()}")
    print("=" * 60)

    # 1. Etiketleri yükle
    try:
        all_entries = load_labels()
    except Exception as e:
        print(f"Hata: Etiket dosyası yüklenemedi: {e}")
        return

    # Sınırlama varsa uygula
    if limit and limit > 0:
        entries = all_entries[:limit]
        print(f"Toplam {len(all_entries)} videodan ilk {limit} tanesi seçildi.")
    else:
        entries = all_entries
        print(f"Toplam {len(entries)} video işlenecek.")

    # 2. Pipeline oluştur
    # VRAM taşmasını önlemek için YOLO'yu CPU'da çalıştırıyoruz
    pipe = build_demo_pipeline(
        mock_vlm=(backend == "mock"),
        vlm_backend=backend,
        yolo_device="cpu",
        load_in_4bit=False,
        use_vllm=use_vllm,

    )
    pipe.save_reports = save_reports
    pipe.store_frame_results = False  # Bellek tasarrufu için tüm kare detaylarını tutma

    results = []
    total_videos = len(entries)
    
    # Metrik sayaçları
    true_positives = 0  # Gerçek kaza tetiklendi
    false_positives = 0 # Rutin sahnede kaza tetiklendi (hata)
    true_negatives = 0  # Rutin sahne tetiklenmedi
    false_negatives = 0 # Gerçek kaza tetiklenmedi (kaçırıldı)
    
    total_processed_frames = 0
    total_vlm_calls = 0
    total_processing_time = 0.0

    for i, entry in enumerate(entries, 1):
        vname = entry["video"]
        label = entry["label"]
        vpath = VIDEOS_DIR / label / vname

        print(f"\n[{i}/{total_videos}] İşleniyor: {label}/{vname}")
        
        if not vpath.is_file():
            print(f"  ⚠️ Video dosyası bulunamadı: {vpath}")
            continue

        is_anomaly = (label != "Normal")
        
        # Video koşumu
        t0 = time.perf_counter()
        pipeline_result = pipe.process_video(
            source=vpath,
            max_frames=max_frames_per_video
        )
        duration = time.perf_counter() - t0
        
        if not pipeline_result.success:
            print(f"  ❌ Hata oluştu: {pipeline_result.error}")
            continue

        frames = pipeline_result.frames_processed
        vlm_calls = pipeline_result.vlm_calls
        # "Tetik" = VLM tehlikeli / orta risk veya event üretti
        if pipeline_result.last_analysis is None:
            triggered = False
        else:
            risk = (pipeline_result.last_analysis.risk or "").lower()
            triggered = risk in (
                "yüksek", "kritik", "high", "critical", "orta", "medium"
            ) or bool(pipeline_result.last_analysis.events)
        
        # Metrik sınıflandırma
        if is_anomaly:
            if triggered:
                true_positives += 1
                status_str = "🟢 DOĞRU TEŞHİS (Kaza Yakalandı)"
            else:
                false_negatives += 1
                status_str = "🔴 KAÇIRILDI (Kaza Tespit Edilemedi)"
        else:
            if triggered:
                false_positives += 1
                status_str = "🟡 YANLIŞ ALARM (Normal Sahnede Tetiklendi)"
            else:
                true_negatives += 1
                status_str = "🟢 DOĞRU TEŞHİS (Normal İzleme)"

        total_processed_frames += frames
        total_vlm_calls += vlm_calls
        total_processing_time += duration

        fps = frames / max(duration, 1e-6)
        savings = (1.0 - (vlm_calls / max(frames, 1))) * 100.0
        
        print(f"  Durum: {status_str}")
        print(f"  Kare: {frames} | VLM@336: {vlm_calls} | Kare başına VLM tasarrufu: %{savings:.0f}")
        print(f"  Hız: {fps:.1f} FPS | Süre: {duration:.2f} s")

        results.append({
            "video": vname,
            "ground_truth_label": label,
            "is_anomaly": is_anomaly,
            "processed_frames": frames,
            "vlm_calls": vlm_calls,
            "triggered": triggered,
            "metric_status": status_str,
            "latency_s": duration,
            "fps": fps,
            "vlm_savings_percent": savings,
            "last_summary": pipeline_result.last_summary
        })

    # 3. İstatistikleri ve Metrikleri Hesapla
    total_runs = len(results)
    if total_runs == 0:
        print("İşlenen video bulunamadı.")
        return

    # Duyarlılık (Recall) ve Kesinlik (Precision)
    recall = true_positives / max(true_positives + false_negatives, 1)
    precision = true_positives / max(true_positives + false_positives, 1)
    f1_score = 2 * (precision * recall) / max(precision + recall, 1e-9)
    accuracy = (true_positives + true_negatives) / total_runs

    avg_vlm_savings = (1.0 - (total_vlm_calls / max(total_processed_frames, 1))) * 100.0
    avg_fps = total_processed_frames / max(total_processing_time, 1e-6)

    # 4. Raporu Konsola Yazdır
    print("\n" + "=" * 60)
    print("BENCHMARK SONUÇLARI ÖZETİ")
    print("=" * 60)
    print(f"Toplam Video Sayısı  : {total_runs}")
    print(f"Toplam Kare Sayısı   : {total_processed_frames}")
    print(f"Toplam VLM@336       : {total_vlm_calls}")
    print(f"Ortalama İşleme Hızı : {avg_fps:.2f} FPS")
    print(f"VLM Tasarruf Oranı   : %{avg_vlm_savings:.1f} (her kare VLM baseline'a göre)")
    print("-" * 60)
    print(f"Doğru Pozitif (TP)   : {true_positives} (Yakaladığımız Kazalar)")
    print(f"Yanlış Negatif (FN)  : {false_negatives} (Kaçırdığımız Kazalar)")
    print(f"Doğru Negatif (TN)   : {true_negatives} (Normal İzlenen Sahneler)")
    print(f"Yanlış Pozitif (FP)  : {false_positives} (Gereksiz Tetiklenen Sahneler)")
    print("-" * 60)
    print(f"Doğruluk (Accuracy)  : %{accuracy * 100:.1f}")
    print(f"Duyarlılık (Recall)  : %{recall * 100:.1f} (Kaza Yakalama Oranı)")
    print(f"Kesinlik (Precision) : %{precision * 100:.1f} (Alarm Güvenilirliği)")
    print(f"F1 Skoru             : {f1_score:.3f}")
    print("=" * 60)

    # 5. Raporu JSON Olarak Kaydet
    report_data = {
        "benchmark_meta": {
            "backend": backend,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "limit": limit,
            "max_frames_per_video": max_frames_per_video
        },
        "metrics": {
            "total_runs": total_runs,
            "total_processed_frames": total_processed_frames,
            "average_fps": avg_fps,
            "vlm_savings_percent": avg_vlm_savings,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1_score": f1_score
        },
        "video_results": results
    }

    OUTPUT_REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_JSON.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetaylı benchmark raporu kaydedildi: {OUTPUT_REPORT_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentinel Benchmark Runner")
    parser.add_argument(
        "--backend",
        type=str,
        default="mock",
        choices=["mock", "smolvlm", "internvl2"],
        help="VLM modeli backend'i (varsayılan: mock)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="İşlenecek maksimum video sayısı (boş bırakılırsa hepsi)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=120,
        help="Video başına işlenecek maks. kare (varsayılan: 120)",
    )
    parser.add_argument(
        "--save-reports",
        action="store_true",
        help="Her video analizi için JSON rapor çıktılarını outputs/reports altına kaydet",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="vLLM motorunu kullan (hızlı çıkarım için)",
    )
    args = parser.parse_args()

    run_benchmark(
        backend=args.backend,
        limit=args.limit,
        max_frames_per_video=args.max_frames,
        save_reports=args.save_reports,
        use_vllm=args.vllm,
    )

