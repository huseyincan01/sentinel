# Sentinel — Kaggle kurulum rehberi

Hedef: **TEK VLM @ 336×336** SmolVLM (veya InternVL2) Kaggle GPU’da çalışsın.

Sık hata:

```text
AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

Bu **VRAM OOM değil**. Anlamı: yüklü PyTorch tekerleği, oturumdaki GPU mimarisi
için CUDA kernel içermiyor (ör. `uv`/pip ile `cu121` torch Kaggle torch’unu ezdi).

---

## Altın kural: torch’a dokunma

| Yap | Yapma |
|-----|--------|
| Kaggle’ın önceden yüklü `torch`’unu kullan | `pip install torch` / `uv sync` ile torch ezmek |
| `requirements-kaggle.txt` kur | Ana `requirements.txt` (içinde torch var) kör kurmak |
| Kernel **Restart** sonrası test | Restart etmeden üst üste torch kurmak |

Proje `pyproject.toml` torch’u **pytorch-cu121** indeksine bağlar — bu **yerel/uv** içindir.
Kaggle notebook’unda `uv sync` **önermeyiz**.

---

## Hücre 0 — GPU oturumu açık mı?

Notebook ayarı: **Accelerator → GPU** (T4 / P100 / L4).

```python
!nvidia-smi
```

---

## Hücre 1 — Torch teşhisi (kurulumdan ÖNCE)

```python
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if torch.cuda.is_available():
    print("capability:", torch.cuda.get_device_capability(0))
    print("arch list:", torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else "n/a")
    x = torch.randn(4, 4, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("matmul OK:", tuple(y.shape))
```

- `matmul OK` → torch sağlıklı; **torch’u bir daha kurma**.
- `no kernel image` → Hücre 3’e (uyumlu torch) geç; sonra **Restart**.

Proje kodu ile:

```python
from src.vlm.cuda_compat import diagnose_cuda, verify_cuda_matmul, humanize_cuda_error
print(diagnose_cuda())
print(verify_cuda_matmul())
```

---

## Hücre 2 — Güvenli paket kurulumu (torch YOK)

Repo kökünde:

```python
# Kaggle: Add data / git clone sonrası repo köküne cd
import os
os.chdir("/kaggle/working/Sentinel")  # kendi yolunu yaz

!pip install -q -r requirements-kaggle.txt
```

`requirements-kaggle.txt` bilerek **torch/torchvision içermez**.

Alternatif (tek satır, torch’suz paketler):

```bash
pip install -q transformers accelerate safetensors sentencepiece einops \
  ultralytics opencv-python-headless scikit-image scipy pydantic pyyaml \
  gradio pillow lm-format-enforcer huggingface_hub httpx
```

---

## Hücre 3 — Sadece matmul patlıyorsa: uyumlu torch (opsiyonel)

**Sadece** Hücre 1 matmul başarısızsa. Sonra **Runtime → Restart session**.

```python
# DİKKAT: yalnızca kernel mismatch doğrulandıysa
!pip uninstall -y torch torchvision torchaudio
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Hâlâ olmazsa cu118 dene:
# !pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Restart sonrası Hücre 1’i tekrarla. Yeşil olmadan model indirme.

---

## Hücre 4 — Hızlı doğrulama (mock → gerçek)

```python
# 1) Mock: CUDA olmadan pipeline
from src.pipeline import build_demo_pipeline
pipe = build_demo_pipeline(mock_vlm=True, vlm_size=336, vlm_period_s=2.0)
print("mock OK")

# 2) CUDA smoke (gerçek VLM öncesi)
from src.vlm.cuda_compat import ensure_cuda_kernels_or_raise
ensure_cuda_kernels_or_raise()
print("CUDA smoke OK")

# 3) UI
# !python -m src.ui.app --share   # CLI destekliyorsa
# veya Gradio notebook içi launch
```

UI checklist:

1. Önce **Mock VLM açık** → analiz akıyor mu?
2. Mock kapalı + **smolvlm** → model yükleme çubuğu “HAZIR” olmalı.
3. CUDA hatası varsa kartta Türkçe rehber görünür (`notebooks/kaggle_setup.md` referansı).

---

## Hücre 5 — Önerilen backend

| GPU (tipik) | Backend | Not |
|-------------|---------|-----|
| T4 16GB | `smolvlm` (varsayılan) | FP16; quant kapalı |
| T4 16GB | `internvl2` | Sunum hedefi; VRAM’e dikkat |
| P100 | `smolvlm` | Eski mimari; torch arch list kontrol |

Quant (4/8-bit) Kaggle’da bitsandbytes ile bazen bozulur; Sentinel varsayılanı **FP16**.

---

## Ne bozar?

1. `uv sync` / `pip install torch==...` ile Kaggle torch’unu ezmek  
2. Yerel Windows tekerleği veya yanlış CUDA major  
3. Restart etmeden yarım kalmış torch kurulumları  
4. Warmup hatasını yok sayıp “ready” sanmak — artık UI **CUDA mismatch’te error** gösterir  

---

## Kod tarafı (bu repoda)

- `src/vlm/cuda_compat.py` — teşhis + Türkçe mesaj  
- Yükleme / `generate` / warmup → `CudaKernelMismatchError`  
- UI model kartı ve VLM hata satırı → okunabilir rehber  

Ham hata yerine şunu görmelisin:

> CUDA mimari uyumsuzluğu: bu PyTorch tekerleği… torch’u yeniden KURMA…
