# Sentinel — Kaggle kurulum rehberi

Hedef: **TEK VLM @ 336×336** SmolVLM (veya InternVL2) Kaggle GPU’da çalışsın.

Sık hata:

```text
AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

Bu **VRAM OOM değil**. Anlamı: yüklü PyTorch tekerleği, oturumdaki GPU’nun
**compute capability**’si için CUDA kernel içermiyor.

---

## Senin log’un ne diyor? (P100 + torch 2.10)

Örnek teşhis:

| Alan | Değer | Yorum |
|------|--------|--------|
| GPU | Tesla **P100** | Pascal, **sm_60** |
| torch | 2.10.0+**cu128** | Kaggle güncel image |
| arch_list | sm_**70**, 75, 80, … | **sm_60 yok** |

→ P100, bu torch ile **hiç çalışmaz**. Sentinel hatası değil; Kaggle torch ↔ eski GPU.

### Çözüm A — Önerilen (en kolay)

1. Notebook **Settings → Accelerator → GPU T4** (P100 değil).  
2. **Restart session**.  
3. Aşağıda Hücre 1 matmul → yeşil olmalı.  
4. `requirements-kaggle.txt` kur; torch’a dokunma.

T4 = sm_75 → güncel torch arch listesinde var.

### Çözüm B — P100’de kalmak zorundaysan

Kaggle’ın torch’unu **bilerek eski sürüme** çek (sm_60 içeren tekerlek):

```python
# Hücre B1 — P100 için torch düşür
!pip uninstall -y torch torchvision torchaudio
!pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

**Runtime → Restart session**, sonra Hücre 1 matmul.

```python
# Hâlâ kırmızıysa cu118 dene:
!pip uninstall -y torch torchvision torchaudio
!pip install -q torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
# Restart
```

> Not: `pip install torch` (sürüm belirtmeden) veya `cu128` en yeni tekerlek
> P100’ü yine kırar. **Sürümlü** kur.

---

## Torch kuralı (güncellenmiş)

| Durum | Ne yap |
|--------|--------|
| Matmul **yeşil** | torch’a **dokunma**; sadece `requirements-kaggle.txt` |
| Matmul kırmızı + **P100/sm_60** | **T4’e geç** veya torch **2.5.x**’e düş |
| Matmul kırmızı + T4 | Bozuk kurulum → cu121 torch yeniden kur + Restart |
| `uv sync` / ana `requirements.txt` | Kaggle’da **kullanma** (torch ezebilir) |

Proje `pyproject.toml` cu121 — **yerel/uv** içindir.

---

## Hücre 0 — GPU oturumu

Notebook: **Accelerator → GPU** — mümkünse **T4**.

```python
!nvidia-smi
```

---

## Hücre 1 — Torch teşhisi

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

- `capability (6, 0)` + arch’ta `sm_60` yok → **Çözüm A veya B**  
- `matmul OK` → torch’u bir daha kurma  

```python
from src.vlm.cuda_compat import diagnose_cuda, verify_cuda_matmul, suggest_cuda_fix_actions
print(diagnose_cuda())
print(verify_cuda_matmul())
print(suggest_cuda_fix_actions())
```

---

## Hücre 2 — Güvenli paketler (torch YOK)

```python
import os
os.chdir("/kaggle/working/Sentinel")  # kendi yolun

!pip install -q -r requirements-kaggle.txt
```

`requirements-kaggle.txt` bilerek **torch/torchvision içermez**.

---

## Hücre 3 — T4’te bozulmuş torch (nadir)

Matmul kırmızı **ve** GPU T4/L4 ise:

```python
!pip uninstall -y torch torchvision torchaudio
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Restart → Hücre 1
```

---

## Hücre 4 — Mock → gerçek VLM

```python
from src.pipeline import build_demo_pipeline
pipe = build_demo_pipeline(mock_vlm=True, vlm_size=336, vlm_period_s=2.0)
print("mock OK")

from src.vlm.cuda_compat import ensure_cuda_kernels_or_raise
ensure_cuda_kernels_or_raise()
print("CUDA smoke OK")
```

UI: önce **Mock VLM**, sonra **smolvlm**.

| GPU | Backend |
|-----|---------|
| T4 16GB | `smolvlm` (varsayılan) veya `internvl2` |
| P100 + eski torch | `smolvlm` FP16 |

Quant (4/8-bit) Kaggle’da bazen bozulur; Sentinel varsayılanı **FP16**.

---

## Kod

- `src/vlm/cuda_compat.py` — teşhis, P100’e özel Türkçe adımlar  
- Warmup CUDA mismatch’te “ready” demez  

Ham `AcceleratorError` yerine UI’da **T4’e geç / torch 2.5.1** rehberi görünür.
