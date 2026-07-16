# SENTINEL — Proje Referans Belgesi
## Bu Dosyayı Okuyan Yapay Zeka İçin Tam Proje Bağlamı

> **ÖNEMLI KURAL — Her AI Ajanı Mutlaka Uymalıdır:**
> Bir part veya faz tamamlandığında yapay zeka ÖNCE o bölüme ait testleri yazar ve çalıştırır.
> Testlerin **tamamı başarıyla geçmedikçe** part tamamlanmış sayılmaz ve bir sonraki parta geçilemez.
> Testler geçtikten sonra "✅ PART X TAMAMLANDI" ilanı yapılır ve progress.md güncellenir.

---

## 1. Proje Kimliği

| Alan | Değer |
|------|-------|
| **Proje Adı** | Sentinel |
| **Tam Açıklama** | Dinamik Hibrit Görsel Algı ve Karar Destek Ajanı |
| **Yarışma** | TEKNOFEST 2026 — Yapay Zeka Kategorisi |
| **Senaryo** | 3. Senaryo: Video Analiz + Karar Destek |
| **Hedef VLM (sunum / Colab)** | InternVL2-8B |
| **Geliştirme VLM (yerel varsayılan)** | **SmolVLM** (`HuggingFaceTB/SmolVLM-Instruct` veya SmolVLM2) |
| **Hedef Donanım** | Geliştirme: RTX 4050 (~6GB) · Sunum/Colab: T4 16GB |
| **Programlama Dili** | Python |
| **Arayüz** | Gradio |

---

## 2. Amaç ve Felsefe

Sentinel endüstriyel kamera görüntülerini analiz ederek tehlikeli olayları tespit eder, bağlamsal yorum üretir, operatöre **Türkçe** aksiyon odaklı rapor sunar ve mock araçlarla simüle müdahale başlatır.

### Temel mimari fikir (TEK VLM)

**Yanlış modeller (kullanma):**
- Her kareyi full-HD VLM’ye vermek
- Ayrı “Gate VLM” + “Detail VLM” / high-res ikinci yol
- `need_high_res` ile zincir tetikleme
- Low-res’i YOLO/MOG2’ye vermek (tracker **high-res** kalır)

**Doğru model:**

```
Kademe 1 — HIGH-RES algı (sürekli, VLM değil)
  YOLO + MOG2 + BoT-SORT + triage
  → tetik? en büyük hareket / aday bölgesi

Kademe 2 — TEK VLM @ 336×336, ~2 sn periyot
  Duvar saati worker; model bitmeden yeni istek yok
  · tetik yok → full frame → 336²
  · tetik var → crop(+%20 dolgu) → 336²
  → AnalysisResult JSON + tools + memory

Yüksek çözünürlüklü ikinci VLM yolu YOK.
Gate / Detail ayrımı YOK.
```

Hedef: **sabit compute bütçesi**; dikkat kırpma ile artar, çözünürlük artmaz.

---

## 2b. Neden eski mimari terk edildi? (AI — hatalardan öğren)

> Bu bölüm **kasıtlı tarihçe + yasak listesidir**. Yeni ajan “3 kademe Gate→Detail’e geri dönelim” dememeli.

### Eski (terk edilen) model

```
Kademe 1: YOLO+MOG2 → aday
Kademe 2: Gate VLM low-res @ “2 Hz” → need_high_res?
Kademe 3: Detail VLM high-res (gate EVET veya bypass) → JSON+tools
```

Amaç doğruydu: pahalı high-res VLM’i seyrek çalıştırmak. **Uygulama ve donanım gerçekliği bozdu.**

### Ne bozuldu? (gözlenen semptomlar)

| Semptom | Kök neden |
|---------|-----------|
| 23 sn videoda Gate ~4 kez; 2 dk’da Gate 0–2 | Gate periyodu bazen **video zamanına** (`idx/fps`) bağlandı; YOLO yavaşken duvarda dakikalar = saniyeler video |
| Gate “periyodik çalışmıyor” | Async’te `need_high_res` → Detail **aynı thread / aynı busy** içinde; Detail bitene kadar Gate ölüyordu |
| Gate 0, Detail 1 | Mantıksal “bağımsız hat” + **tek GPU / tek model** + kilit / ilk-infer takılması |
| UI Gate sayacı yalan | Frame yield anında `fr.gate_called` hâlâ false; async sayaç okunmuyordu |
| Mock her seferinde escalate | Sistem promptundaki “kaza/devril” kelimeleri mock danger listesine takılıyordu |
| “2 Hz” iddiası 4050’de imkânsız | SmolVLM tek kare ~1–5+ sn; frekans tavanı donanım, kod değil |

### Mimari hatalar (tekrarlama)

1. **İki VLM rolü, bir model, bir VRAM**  
   “Gate ↛ Detail, Detail ↛ Gate” mantığı kağıtta bağımsız; pratikte `generate` serileşir. İkinci rol (high-res Detail) birincisini (periyodik Gate) boğar.

2. **Zincir tetik (`need_high_res` → Detail)**  
   Hakem her “evet” dediğinde pahalı yol açılır; hakem yavaşsa veya hep escalate ederse sistem ya kilitlenir ya spam üretir.

3. **İddialı frekans vaadi (2 Hz)**  
   Ürün cümlesi donanımdan kopuksa jüri/kullanıcı “bozuk” der. Dürüst söz: **hedef periyot + model bitmeden yeni istek yok**.

4. **Video zamanı ≠ duvar saati**  
   Canlı UI’da kare işleme yavaşsa `timestamp = frame/fps` ile frekans ölçmek Gate’i yok eder. Periyot **duvar saati** olmalı.

5. **Busy flag birleştirme / yanlış scope**  
   `_gate_busy` Detail süresince true kalırsa periyodik gözetim ölür. Tek yol varsa tek busy yeter.

6. **UI’nın async gerçeğini yansıtmaması**  
   Sayaç/overlay pipeline state’ten okunmalı (`_vlm_call_count`), yield anındaki `FrameResult` alanından değil.

7. **Dokümantasyon / kod drift**  
   AGENTS “3 kademe” derken kod “bağımsız” veya “tek VLM” olunca sonraki AI eski şemayı yeniden yazar. **AGENTS = çalışan mimari.**

### Neden şimdiki model?

| Eski | Yeni |
|------|------|
| 2 VLM yolu (gate + detail) | **1 VLM yolu** |
| High-res ara sıra | High-res VLM **hiç yok** |
| Gate frekansı + Detail cooldown karmaşası | Tek ritim: **~2 sn**, model bitsin |
| Tetik = hangi VLM? | Tetik = **crop mı full mü** (ikisi de 336) |
| Tasarruf “Detail’i azalt” | Tasarruf “**her kare full VLM yok** + sabit 336” |

**Ürün cümlesi (jüri):**  
*Pahalı VLM’i her karede full çözünürlükte yakmıyoruz. Sabit 336 bütçeyle ~2 sn’de bir bakıyoruz; hareket olunca aynı bütçeyle ilgi bölgesine zoom’luyoruz — çözünürlük değil dikkat artıyor. YOLO/MOG2 high-res algı katmanı.*

### Yasak listesi (yeni kod yazarken)

- [ ] Gate + Detail iki worker / iki escalate yolu geri getirme  
- [ ] `need_high_res` ile high-res VLM açma  
- [ ] VLM girdisini 1024×768 “detail” yapmak  
- [ ] Periyodu yalnızca `frame_idx/fps` ile sınırlamak (UI async)  
- [ ] Model bitmeden üst üste VLM kuyruğu (kullanıcı: *model çalışsın*)  
- [ ] Low-res frame’i YOLO/MOG2’ye vermek  

### Hâlâ kodda durabilen “ölü” parçalar (bilinçli)

`analyze_gate`, `GateDecision`, `gate_size`/`gate_hz` alias’ları — **test / geriye uyum**. Pipeline ürün yolu bunları kullanmaz. Yeni özellik **`vlm_size` / `vlm_period_s` / `start_vlm_worker` / `analyze_detail@336`** üzerine yazılır.

---

## 3. Sistem Mimarisi

```
[Video]
   │
   ├─► [Ring buffer]  (track / rapor için)
   │
   ▼
[Kademe 1] HybridTracker HIGH-RES
   YOLO + MOG2 + triage
   │
   ▼ her kare: snapshot hazırla
   full→336  VEYA  crop(+%20)→336
   │
   ▼
[Tek VLM worker]  period ≈ 2.0 sn, model bitsin
   analyze_detail (tek şema) → reports/
   │
   ▼
[Gradio] Türkçe özet + JSON + VLM@336 sayacı
```

### Rol tablosu

| Bileşen | Çözünürlük | Ne zaman | İş |
|---------|------------|----------|-----|
| YOLO + BoT-SORT | **Yüksek** | Sürekli | Nesne, ID, giriş, hız/stillness |
| MOG2 | **Yüksek** | Sürekli | Şekilsiz hareket, crop kutusu |
| SSIM / ROI / color_fire | — | Triage | Aday / kırpma sinyali |
| **Tek VLM** | **336×336** | ~2 sn periyot | Tehlike JSON + tools |

---

## 4. Geliştirme Ortamı

| | |
|--|--|
| Lokal | Kod, mock VLM, 4050’de 4-bit deneme |
| Colab T4 | Gerçek InternVL2 8-bit |
| Sunum | Tamamen yerel, harici API yok |
| Bağımlılık | **Birincil:** `uv` + `pyproject.toml` + `uv.lock`. **Ayna:** `requirements.txt`. |

### VLM backend

| Backend | Model ID (varsayılan) | Ne zaman |
|---------|----------------------|----------|
| **`smolvlm`** | `HuggingFaceTB/SmolVLM-Instruct` | Yerel 4050 — varsayılan |
| **`internvl2`** | `OpenGVLab/InternVL2-8B` | Colab / güçlü PC |
| **`mock`** | — | CI / test |

- Pipeline **backend’den bağımsızdır** (tek `AnalysisResult` şeması).
- Jüri: *“Hedef InternVL2-8B; geliştirme SmolVLM ile doğrulandı.”*

### VRAM
- **4050 ~6GB + SmolVLM:** 4-bit önerilir; YOLO **CPU**  
- **T4 + InternVL2-8B:** 8-bit; OOM → 4-bit  
- VLM girdisi **her zaman 336×336**

---

## 5. Teknolojiler

| Teknoloji | Amaç |
|-----------|------|
| ultralytics YOLOv8 | High-res tespit |
| OpenCV MOG2 | High-res hareket / crop |
| BoT-SORT / SimpleIoU | Track; MOG2 ReID kapalı |
| scikit-image SSIM | Ani değişim adayı |
| transformers + **SmolVLM** | Yerel tek VLM |
| transformers + **InternVL2-8B** | Hedef / Colab |
| lm-format-enforcer + pydantic | JSON |
| gradio | UI |

---

## 6. Klasör Yapısı

```
Sentinel/
├── src/
│   ├── tracking/     hybrid_tracker, mog2, botsort
│   ├── decision/     triage_engine (aday + cooldown + crop sinyali)
│   ├── vlm/          internvl_agent, tools, schemas, memory
│   ├── pipeline/     main_pipeline (track + tek VLM worker)
│   └── ui/           app.py
├── tests/
├── pyproject.toml
├── uv.lock
├── requirements.txt
├── AGENTS.md
├── progress.md
└── README.md
```

---

## 7. Çıktı JSON — Tek VLM (şartname)

```json
{
  "summary": "Videoda forklift kazası ve yaralanma riski gözlenmiştir.",
  "events": [
    {"time": "00:15", "event": "Forklift devrildi", "severity": "Yüksek"},
    {"time": "00:20", "event": "Yerde hareketsiz kişi", "severity": "Kritik"}
  ],
  "risk": "Yüksek",
  "risk_score": 0.87,
  "actions": ["Sağlık ekibini çağır", "Alanı güvenlik altına al"],
  "tools_called": ["call_ambulance", "lock_area"],
  "timestamp": "2026-07-15T00:10:00",
  "frame_analyzed": 450
}
```

`GateDecision` / `need_high_res` **pipeline’da kullanılmaz** (şema kodda kalabilir; ölü yol).

---

## 8. Mock Araçlar

`call_ambulance`, `alert_security_team`, `lock_area`, `generate_incident_report`, `notify_supervisor`, `trigger_alarm`  
→ tek VLM analizi sonrası, simülasyon.

---

## 9. Bellek (Memory)

- Kayan pencere N=10, sticky M=5 (Yüksek/Kritik)  
- Tek VLM prompt’una eklenir  

---

## 10. Aday sinyalleri (Kademe 1 → kırpma / bağlam)

Bunlar **ikinci bir VLM yolu açmaz**. Periyodik 336 VLM zaten çalışır; tetik varsa **crop** tercih edilir.

| Sinyal | Kaynak | Kural (varsayılan) |
|--------|--------|---------------------|
| **Stillness** | YOLO track | person, hız≈0, ≥3.5 sn |
| **Dangerous motion** | YOLO track | bbox sıçrama / yüksek hız |
| **Entrance** | YOLO | yeni person/araç ID |
| **MOG2 motion** | MOG2 | güçlü blob |
| **SSIM shock** | SSIM | skor &lt; 0.85 (şiddetli &lt; 0.50) |
| **ROI** | opsiyonel poligon | nesne içinde |
| **Periodic** | zaman | triage heartbeat (crop zorunlu değil) |
| **Color fire** | HSV | kırmızı/turuncu/sarı ≥150 px |

### Frekans

| Katman | Limit |
|--------|--------|
| Tek VLM | **~2.0 sn** periyot; **model bitmeden yeni istek yok** |
| Triage coalesce / cooldown | Detail spam’i değil; rapor/alert yorgunluğu için |

---

## 11. BoT-SORT / ID

- YOLO: ReID mümkünse açık  
- MOG2: **ReID kapalı**  
- Prefix: `yolo_`, `mog_`

---

## 12. VLM implementasyon notları

1. Tek şema: **AnalysisResult** (tools + memory)  
2. Girdi: **her zaman 336×336** (`downscale_for_vlm`)  
3. Crop: MOG2 en büyük kontur veya track birleşimi + **%20 padding** → resize 336  
4. Mock: `make_mock_generator`  
5. Fabrika: `create_vlm_agent(backend="smolvlm"|"internvl2"|"mock")`  
6. **Global model cache** + Gradio **eager load / CUDA warmup**  
7. UI: `start_vlm_worker()` / `stop_vlm_worker()` — arka plan thread, video akışı donmasın  
8. Sayaç: `_vlm_call_count` (UI: `VLM@336`)  
9. SmolVLM: `repetition_penalty=1.3`, JSON prefix injection `{`  
10. InternVL2: mümkünse `generate()` + format enforcer; `_prepare_image` always-true yasak  

---

## 13. KPI

| KPI | Hedef |
|-----|--------|
| VLM girdi boyutu sabit | 336×336 |
| Periyot | ~2 sn (model uzarsa periyot uzar) |
| Kare başına VLM oranı | düşük (ör. 30 FPS’te ≪ her kare) |
| JSON geçerliliği | %98+ |
| Kritik yakalama | &gt; %90 |

---

## 14. TEKNOFEST kriterleri

Fonksiyonellik %35 · Teknik mimari %35 · Otonomi %20 · Yenilik %10  
Yerel, Türkçe, JSON+özet, tools, multimodal.

---

## 15. Genel kurallar

1. Test zorunluluğu + progress.md  
2. Türkçe yorumlar kritik bloklarda  
3. **Bağımlılık:** `pyproject.toml` / `uv.lock` birincil; `requirements.txt` ayna. Python **>=3.12**.  
4. Büyük hata → yeni PART şişirme; `progress.md` revizyon günlüğü.  
5. **Mimari sadakat:**  
   - YOLO/MOG2 **high-res** (VLM değil)  
   - **Tek VLM** 336, ~2 sn, model bitsin  
   - High-res Detail / Gate zinciri **yok**  
6. Circular import yok  

---

## 16. Pipeline API özeti

```python
# process_frame (UI async):
# 1. buffer.append(frame)
# 2. tracks = tracker.process_frame(high_res)
# 3. decision = triage.evaluate(...)
# 4. img336, cropped = full→336  OR  crop(+20%)→336
# 5. publish_snapshot(img336)
#
# VLM worker (ayrı thread, ~2 sn):
#    snap al → agent.analyze_detail(336) → bitene kadar bekle
#    sleep(max(0, period - elapsed))
```

Parametreler: `vlm_size=336`, `vlm_period_s=2.0`.  
Eski isimler (`gate_size`, `gate_hz`, `start_gate_worker`) yalnızca **alias**; yeni kod `vlm_*` kullanmalı.
