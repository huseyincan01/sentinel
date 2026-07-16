# Sentinel - Progress Tracker

> **Kurallar:**
> 1. Güncel mimari ve teknik kaynak: **`AGENTS.md`** (tek doğru referans).
> 2. Part bitince testler yeşil olmalı; tamamlanan part `[x]` kalır.
> 3. **Büyük hata / yön değişimi → yeni PART ekleme; mevcut parta “Revizyon günlüğü” satırı yaz.**
> 4. Bağımlılık: birincil **`uv` + `pyproject.toml`**; pip/Colab için **`requirements.txt`**.

**Python:** `>=3.12` (hedef 3.12)

**Güncel mimari (özet):**
1. High-res YOLO + MOG2 + track → tetik / kırpma  
2. **Tek VLM** 336×336, **~2 sn** periyot (model bitsin)  
3. Gate / Detail / high-res ikinci VLM **yok**  
**VLM:** geliştirme `smolvlm` · hedef `internvl2-8B` · test `mock`

---

## PART 1: Tracking (YOLO + MOG2 + BoT-SORT)
**Durum:** `[x] Tamamlandı` — 2026-07-15

Klasör, uv, hybrid tracker, `yolo_`/`mog_` prefix, ReID politikası, `test_part1_tracking.py`.

### Revizyon günlüğü
| Tarih | Not |
|-------|-----|
| 2026-07-15 | İlk teslim — 25 test |

---

## PART 2: Triage Engine
**Durum:** `[x] Tamamlandı` — 2026-07-15

ROI/SSIM/periyodik, coalescing, adaptif cooldown. `test_part2_triage.py`.

### Revizyon günlüğü
| Tarih | Hata / karar | Düzeltme |
|-------|----------------|----------|
| 2026-07-15 | ROI zorunlu sanıldı | ROI **opsiyonel** |
| 2026-07-15 | Tetik = full VLM | Tetik = **kırpma/bağlam** (tek VLM periyodik) |

---

## PART 3: VLM + Tools + Memory
**Durum:** `[x] Tamamlandı` — 2026-07-15

Tools, schemas, memory, agent. `test_part3_vlm.py`.

### Revizyon günlüğü
| Tarih | Hata / karar | Düzeltme |
|-------|----------------|----------|
| 2026-07-15 | `_prepare_image` `or True` | Kaldırıldı |
| 2026-07-15 | JSON parse | pre-validator + prefix injection |
| 2026-07-15 | Gate+Detail iki şema | **Pipeline tek şema:** AnalysisResult @336 |
| 2026-07-15 | Backend | `smolvlm` \| `internvl2` \| `mock` |

---

## PART 4: Pipeline entegrasyonu
**Durum:** `[x] Tamamlandı` — 2026-07-15

Video döngüsü, track → VLM → tools → reports. `test_part4_pipeline.py`.

### Revizyon günlüğü
| Tarih | Hata / karar | Düzeltme |
|-------|----------------|----------|
| 2026-07-15 | Dual-res gate/detail | **Tek VLM 336 / 2 sn** |
| 2026-07-15 | Tracker low-res | Tracker **her zaman high-res** |

---

## PART 5: Gradio + yerel checklist
**Durum:** `[x] Tamamlandı` — 2026-07-15

UI, README, `test_part5_pipeline_ui.py`.

### Revizyon günlüğü
| Tarih | Hata / karar | Düzeltme |
|-------|----------------|----------|
| 2026-07-15 | Gate/Detail UI | **VLM@336** tek sayaç / tek durum |
| 2026-07-15 | Async + 2 Hz gate | **start_vlm_worker**, period 2 sn, model bitsin |
| 2026-07-15 | High-res detail panelleri | Kaldırıldı; crop bilgisi gösterilir |
| 2026-07-15 | Cache / warmup / 4-bit / color fire / temporal crop | Korundu (crop artık 336’ya gider) |
| 2026-07-16 | Model yükleme + sürekli çalışma | UI: **global model yükleme çubuğu** (hazır olmadan Analiz kapalı); VLM **tek sürekli worker döngüsü** (bitince hemen sonraki snapshot, min ~2 sn); video boyunca worker açık, model her analizde yeniden yüklenmez. |
| 2026-07-16 | ROI yan paneli karmaşası | ROI kavramı tamamen kaldırıldı. Arayüzdeki iki panel birleştirilerek tek bir **VLM ve Tetikleyici/Odak Durumu** kartı yapıldı. Testler 5 dönüş elemanına göre güncellendi. |
| 2026-07-16 | Kaggle/Colab entegrasyonu ve Hız | `main.py` ve `benchmark.py` dosyalarına `--share` ve `--vllm` CLI argümanları eklendi. Bulut ortamlarında çıkarımı 5-10 kat hızlandırmak için **vLLM** entegre edildi. Windows/GPU-suz ortamlarda otomatik `transformers` kütüphanesine güvenli fallback sağlandı. |
| 2026-07-16 | UI Düzenlemeleri & Geribildirim | VLM'nin işlediği 336x336 karesi canlı önizlemenin hemen altına yerleştirildi. Riskin seviyesine göre (kırmızı/yeşil/turuncu/gri) anlık ince çerçeve çizimi eklendi. Türkçe özet en alta genişletildi. Model yükleme durumundaki titreme/yanıp sönme (flashing) önbellek ile çözüldü. Mock model kilitlenmesi giderildi. |



---

## 🤖 HANDOVER (sonraki AI)

**Önce oku:** `AGENTS.md` §2b — *Neden eski mimari terk edildi?* (Gate→Detail hataları, semptomlar, yasak listesi).

**Güncel mimari — BOZMAYIN:**
```
YOLO+MOG2 (high-res) → tetik/kırpma
     │
     ▼ ~2 sn (duvar; model bitene kadar)
Tek VLM @ 336×336  (full veya crop+%20)
     → AnalysisResult + tools
```

- UI: `start_vlm_worker()` / `stop_vlm_worker()` · sayaç `_vlm_call_count`
- Param: `vlm_size=336`, `vlm_period_s=2.0`
- Eski `gate_*` / `analyze_gate` alias/ölü API — **yeni özellik yazma**
- Kaynak: **AGENTS.md**, **README.md**

Açık iş: kırpma isabeti + SmolVLM tehlike kalitesi.

---

## Neden Gate/Detail terk edildi? (kısa — detay AGENTS §2b)

Kronolojik öğrenme (2026-07-15):

1. **3 kademe (track → gate@2Hz → detail high-res)** yazıldı; hedef tasarruf doğruydu.  
2. **Gate, Detail’e bağlı** (`need_high_res`) + async busy hataları → periyodik gözetim fiilen öldü.  
3. **“Bağımsız Gate/Detail”** denendi; tek SmolVLM + VRAM hâlâ serileştiriyordu; 2 dk’da Gate 0 görüldü.  
4. **Periyot video zamanına** bağlanınca yavaş YOLO = az Gate (duvar ≠ video).  
5. Kullanıcı kararı: high-res Detail ve çift tetik panelini **yok et**; **tek 336, ~2 sn, model bitsin**; tetik sadece **crop**.  

**Tek cümle:** İki rol + bir yavaş model + iddialı 2 Hz = vaat ile gerçek kopuk; sabit 336 + dürüst 2 sn periyot = ölçülebilir ve anlatılabilir.

---

## Revizyon özeti (öğrenilenler)

| # | Yanlış / denenen | Doğru (şimdi) | Ders |
|---|------------------|---------------|------|
| 1 | Low-res = YOLO katmanı | YOLO/MOG2 **high-res**; 336 yalnız VLM | Algı ile VLM bütçesini ayır |
| 2 | Gate → Detail zinciri | **Tek VLM**, zincir yok | İkinci yol birincisini boğar |
| 3 | High-res Detail “bazen” | High-res VLM **yok** | “Bazen” async’te her zaman pahalıya döner |
| 4 | Bağımsız Gate@2Hz vaadi | ~**2 sn**, model bitsin | Frekans = min(hedef, 1/infer) |
| 5 | Periyot = video `t=f/fps` | Periyot = **duvar saati** (UI) | Yavaş track Gate’i öldürür |
| 6 | UI `fr.gate_called` | `_vlm_call_count` | Async’te FrameResult yalan söyleyebilir |
| 7 | Mock danger = full prompt | Yalnız aday satırı | Sistem prompt false positive |
| 8 | Sürekli full-HD VLM | Sabit 336 + crop | Dikkat ≠ megapiksel |
| 9 | Her hatada yeni PART | **Revizyon günlüğü** | Tarihçe kaybolmasın |

---

## Bağımlılık

| Dosya | Rol |
|--------|-----|
| `pyproject.toml` + `uv.lock` | Birincil (uv) |
| `requirements.txt` | pip ayna |

Kurulum: `uv sync` veya `pip install -r requirements.txt` · Python **>=3.12**
