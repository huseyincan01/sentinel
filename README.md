# Sentinel

**Dinamik Hibrit Görsel Algı ve Karar Destek Ajanı**  
TEKNOFEST 2026 — Yapay Zeka · Senaryo 3: Video Analiz + Karar Destek

Sentinel, endüstriyel kamera görüntülerini analiz ederek tehlikeli olayları tespit eder, bağlamsal yorum üretir, operatöre **Türkçe** aksiyon odaklı rapor sunar ve mock araçlarla simüle müdahale başlatır.

## Temel fikir

Sentinel **iki katmanlı** çalışır (tek VLM):

1. **High-res** YOLO + MOG2 + BoT-SORT → nesne/hareket, tetik, kırpma kutusu  
2. **Tek VLM** @ **336×336**, yaklaşık **2 sn**’de bir (model bitmeden yeni istek yok)  
   - Tetik yok → full kare → 336  
   - Tetik var → en büyük değişim + **%20 dolgu** → 336  

**Yok:** ayrı Gate VLM, high-res Detail VLM, `need_high_res` zinciri.

```
Video → high-res track → (opsiyonel crop) → 336 VLM @ ~2s → JSON + tools
```

### Neden 3 kademe (Gate→Detail) yok?

Kısa: RTX 4050 + SmolVLM üzerinde Gate@2Hz + high-res Detail **aynı modelde** birbirini kilitledi; “2 Hz” vaadi ölçülemedi (Gate 0–4 / video).  
Şimdi: **tek bütçe (336), dürüst periyot (~2 sn, model bitsin), tetik = crop**.  
Ayrıntılı hata günlüğü ve yasak listesi: **`AGENTS.md` §2b**.

## Özellikler

| Katman | İçerik |
|--------|--------|
| Track | High-res YOLOv8 + MOG2 + BoT-SORT |
| VLM | Tek model · **336×336** · ~**2 sn** periyot |
| Geliştirme VLM | **SmolVLM** (4050) · Hedef: InternVL2-8B |
| Adaylar | SSIM, stillness, motion, entrance, MOG2, color_fire, opsiyonel ROI → **kırpma** |
| UI | Gradio · `VLM@336` sayacı |

## Gereksinimler

- Python **3.12+**
- **`uv sync`** (önerilen) veya `pip install -r requirements.txt`
- İsteğe bağlı: NVIDIA GPU (gerçek VLM; mock ile CPU demo)

**Şartname:** Harici API yok. Tüm inference yereldir.

## Kurulum

```bash
uv sync
uv run pytest tests/ -q
```

### Kaggle / Colab

**Torch’u yeniden kurma.** Kaggle’ın ön yüklü CUDA torch’unu kullan; aksi halde sık görülen hata:

`CUDA error: no kernel image is available for execution on the device`

| Dosya | Ne işe yarar |
|-------|----------------|
| `requirements-kaggle.txt` | torch **olmadan** paketler |
| `notebooks/kaggle_setup.md` | Teşhis hücreleri + mock→smolvlm checklist |

```bash
pip install -r requirements-kaggle.txt   # Kaggle
# Ana requirements.txt yerel/uv içindir (torch içerir)
```

Kod: `src/vlm/cuda_compat.py` — yükleme/warmup/inferansta Türkçe mimari uyumsuzluk mesajı.

## Çalıştırma

### Gradio

```bash
uv run python -m src.ui.app
```

`http://127.0.0.1:7860`

- Video yükle → **Analizi Başlat**
- Mock VLM: model indirmeden demo  
- Gerçek model: Mock kapalı + **smolvlm** (4050) veya **internvl2**

### Programatik

```python
from src.pipeline import build_demo_pipeline

pipe = build_demo_pipeline(mock_vlm=True, vlm_size=336, vlm_period_s=2.0)
result = pipe.process_video("data/test_videos/sample_test.mp4")
print(result.last_summary, result.vlm_calls)
```

## Proje yapısı

```
Sentinel/
├── src/
│   ├── tracking/
│   ├── decision/      # triage
│   ├── vlm/           # agent, tools, memory, schemas
│   ├── pipeline/      # tek VLM orkestrasyon
│   └── ui/
├── tests/
├── outputs/reports/
├── AGENTS.md          # mimari kaynak (AI + insan)
├── progress.md
├── pyproject.toml
└── requirements.txt
```

## Çıktı JSON

```json
{
  "summary": "Videoda forklift kazası ve yaralanma riski gözlenmiştir.",
  "events": [
    {"time": "00:15", "event": "Forklift devrildi", "severity": "Yüksek"}
  ],
  "risk": "Yüksek",
  "risk_score": 0.87,
  "actions": ["Sağlık ekibini çağır", "Alanı güvenlik altına al"],
  "tools_called": ["call_ambulance", "lock_area"],
  "timestamp": "2026-07-15T00:10:00",
  "frame_analyzed": 450
}
```

## Mock araçlar

- `call_ambulance` · `alert_security_team` · `lock_area`  
- `generate_incident_report` · `notify_supervisor` · `trigger_alarm`

## Geliştirme notları

| Ortam | Kullanım |
|-------|----------|
| Lokal (4050) | SmolVLM 4-bit + YOLO CPU |
| Sunum | InternVL2-8B veya SmolVLM, tamamen yerel |

## Testler

```bash
uv run pytest tests/ -q
```

## Lisans / yarışma

TEKNOFEST 2026 — `ultralytics`, `opencv`, `transformers`, `gradio`, vb.
