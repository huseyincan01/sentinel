"""
CUDA / PyTorch uyumluluk teşhisi.

Kaggle/Colab'da sık görülen:
  AcceleratorError: CUDA error: no kernel image is available for execution on the device

Anlamı: yüklü torch tekerleği, oturumdaki GPU compute capability için kernel içermiyor.
VRAM OOM veya model indirme hatası DEĞİL.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Union


# Ham hata metninde aranacak imzalar (case-insensitive)
_KERNEL_MISMATCH_PATTERNS = (
    r"no kernel image is available for execution on the device",
    r"cudaerrornokernelimage",
    r"cuda error: no kernel image",
    r"error no kernel image",
)

_KERNEL_RE = re.compile("|".join(_KERNEL_MISMATCH_PATTERNS), re.IGNORECASE)


class CudaKernelMismatchError(RuntimeError):
    """Torch CUDA binary ↔ GPU mimarisi uyumsuzluğu."""

    def __init__(self, message: str, original: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.original = original


def is_cuda_kernel_mismatch(
    exc_or_text: Union[BaseException, str, None],
) -> bool:
    """Exception veya metin CUDA 'no kernel image' hatası mı?"""
    if exc_or_text is None:
        return False
    if isinstance(exc_or_text, CudaKernelMismatchError):
        return True
    text = str(exc_or_text)
    if _KERNEL_RE.search(text):
        return True
    # Bazı ortamlarda yalnızca type adı + kısa mesaj
    if isinstance(exc_or_text, BaseException):
        name = type(exc_or_text).__name__.lower()
        if "accelerator" in name and "kernel" in text.lower():
            return True
        # Zincirlenmiş neden
        cause = getattr(exc_or_text, "__cause__", None) or getattr(
            exc_or_text, "__context__", None
        )
        if cause is not None and cause is not exc_or_text:
            if is_cuda_kernel_mismatch(cause):
                return True
    return False


def diagnose_cuda() -> Dict[str, Any]:
    """
    Ortam özeti (torch yoksa güvenli dict).
    GPU üzerinde gerçek kernel denemesi YAPMAZ — sadece metadata.
    """
    info: Dict[str, Any] = {
        "torch_version": None,
        "torch_cuda_built": None,
        "cuda_available": False,
        "device_name": None,
        "capability": None,
        "arch_list": None,
        "capability_supported": None,
        "note": "",
    }
    try:
        import torch
    except ImportError:
        info["note"] = "torch yüklü değil"
        return info

    info["torch_version"] = getattr(torch, "__version__", "?")
    info["torch_cuda_built"] = getattr(getattr(torch, "version", None), "cuda", None)
    info["cuda_available"] = bool(torch.cuda.is_available())
    if not info["cuda_available"]:
        info["note"] = "CUDA kullanılamıyor (CPU torch veya sürücü yok)"
        return info

    try:
        info["device_name"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        info["capability"] = f"sm_{cap[0]}{cap[1]}"
        info["capability_tuple"] = (int(cap[0]), int(cap[1]))
    except Exception as exc:  # pragma: no cover
        info["note"] = f"cihaz okunamadı: {exc}"
        return info

    arch_list = None
    try:
        if hasattr(torch.cuda, "get_arch_list"):
            arch_list = list(torch.cuda.get_arch_list())
    except Exception:
        arch_list = None
    info["arch_list"] = arch_list

    if arch_list and info.get("capability"):
        sm = info["capability"]  # sm_75
        sm_num = sm.replace("sm_", "")
        # arch list genelde ['sm_50', 'sm_60', ...] veya compute_XY
        supported = any(
            re.search(rf"(sm_|compute_){re.escape(sm_num)}\b", str(a))
            for a in arch_list
        )
        info["capability_supported"] = supported
        if not supported:
            info["note"] = (
                f"GPU {sm} torch.arch_list içinde yok — kernel mismatch riski yüksek"
            )
    return info


def verify_cuda_matmul() -> Dict[str, Any]:
    """
    Küçük matmul ile gerçek CUDA kernel testi.
    Returns: ok, error, diagnosis
    """
    diag = diagnose_cuda()
    out: Dict[str, Any] = {"ok": False, "error": None, "diagnosis": diag}
    if not diag.get("cuda_available"):
        out["error"] = diag.get("note") or "CUDA yok"
        return out
    try:
        import torch

        x = torch.randn(4, 4, device="cuda", dtype=torch.float16)
        y = x @ x
        torch.cuda.synchronize()
        _ = float(y.sum().item())
        out["ok"] = True
        return out
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out


def _capability_major_minor(cap: Optional[str]) -> Optional[tuple]:
    """'sm_60' / 'sm_75' → (6, 0) / (7, 5)."""
    if not cap:
        return None
    m = re.search(r"sm_(\d+)", str(cap))
    if not m:
        return None
    num = m.group(1)
    if len(num) == 2:
        return int(num[0]), int(num[1])
    if len(num) >= 3:
        # sm_100, sm_120
        return int(num[:-1]), int(num[-1])
    return int(num), 0


def suggest_cuda_fix_actions(diagnosis: Optional[Dict[str, Any]] = None) -> list:
    """
    Teşhise göre somut adımlar (sıralı öncelik).

    Kaggle 2025–2026: varsayılan torch (ör. 2.10+cu128) sıkça sm_70+;
    Tesla P100 (sm_60) bu tekerleklerde YOK → 'torch'a dokunma' yanlış tavsiye.
    """
    diag = diagnosis if diagnosis is not None else diagnose_cuda()
    name = (diag.get("device_name") or "").lower()
    cap = diag.get("capability") or ""
    mm = _capability_major_minor(cap)
    supported = diag.get("capability_supported")
    actions: list = []

    # P100 / Pascal (sm_60, sm_61): modern Kaggle torch desteklemiyor
    is_pascal = bool(mm and mm[0] == 6) or "p100" in name or "pascal" in name
    if is_pascal or (supported is False and mm and (mm[0] < 7)):
        actions.append(
            "ÖNERİLEN: Kaggle notebook Settings → Accelerator = GPU T4 "
            "(sm_75; güncel torch ile uyumlu). P100 (sm_60) yeni PyTorch tekerleklerinde yok."
        )
        actions.append(
            "P100’de kalacaksan: Runtime Restart sonrası ESKİ torch kur "
            "(sm_60 içeren, ör. 2.5.x+cu121): "
            "pip uninstall -y torch torchvision torchaudio; "
            "pip install torch==2.5.1 torchvision==0.20.1 "
            "--index-url https://download.pytorch.org/whl/cu121 "
            "→ tekrar Restart → matmul smoke."
        )
        actions.append(
            "Kaggle’ın varsayılan torch 2.10+cu128 P100’de ÇALIŞMAZ "
            "(arch_list sm_70+). 'Torch kurma' kuralı yalnızca matmul yeşilken geçerlidir."
        )
        return actions

    if supported is False:
        actions.append(
            "GPU capability, yüklü torch arch_list’te yok. "
            "T4/L4 oturumu dene veya notebooks/kaggle_setup.md ‘uyumlu torch’ hücresi."
        )
        actions.append(
            "pip uninstall -y torch torchvision torchaudio; "
            "pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu121 → Kernel Restart."
        )
        return actions

    # capability listede ama yine kernel hatası (nadir / bozulmuş kurulum)
    actions.append(
        "Kernel Restart. Matmul yeşilse torch’u yeniden KURMA; "
        "requirements-kaggle.txt kullan (torch yok)."
    )
    actions.append(
        "Matmul hâlâ kırmızıysa: notebooks/kaggle_setup.md Hücre 3 (uyumlu torch)."
    )
    return actions


def format_cuda_kernel_mismatch_message(
    original: Union[BaseException, str, None] = None,
    diagnosis: Optional[Dict[str, Any]] = None,
) -> str:
    """UI / log için Türkçe, aksiyon odaklı mesaj."""
    diag = diagnosis if diagnosis is not None else diagnose_cuda()
    lines = [
        "CUDA mimari uyumsuzluğu: bu PyTorch tekerleği, oturumdaki GPU için kernel içermiyor "
        "(no kernel image / cudaErrorNoKernelImage).",
        "Bu VRAM bitmesi veya model indirme hatası DEĞİL.",
    ]
    bits = []
    if diag.get("device_name"):
        bits.append(f"GPU={diag['device_name']}")
    if diag.get("capability"):
        bits.append(f"capability={diag['capability']}")
    if diag.get("torch_version"):
        bits.append(f"torch={diag['torch_version']}")
    if diag.get("torch_cuda_built"):
        bits.append(f"torch.cuda={diag['torch_cuda_built']}")
    if diag.get("arch_list") is not None:
        bits.append(f"arch_list={diag['arch_list']}")
    if bits:
        lines.append("Ortam: " + " | ".join(bits))
    if diag.get("capability_supported") is False:
        lines.append(
            "Uyarı: GPU compute capability, yüklü torch arch listesinde görünmüyor."
        )
    actions = suggest_cuda_fix_actions(diag)
    lines.append("Ne yapmalı:")
    for i, step in enumerate(actions, 1):
        lines.append(f"  ({i}) {step}")
    lines.append("Ayrıntı: notebooks/kaggle_setup.md (P100 / T4 bölümü).")
    if original is not None:
        raw = str(original)
        if len(raw) > 280:
            raw = raw[:280] + "…"
        lines.append(f"Ham hata: {raw}")
    return "\n".join(lines)


def raise_if_cuda_kernel_mismatch(
    exc: BaseException,
    *,
    diagnosis: Optional[Dict[str, Any]] = None,
) -> None:
    """Uyum hatasıysa CudaKernelMismatchError fırlat; değilse no-op."""
    if not is_cuda_kernel_mismatch(exc):
        return
    msg = format_cuda_kernel_mismatch_message(original=exc, diagnosis=diagnosis)
    raise CudaKernelMismatchError(msg, original=exc) from exc


def humanize_cuda_error(exc_or_text: Union[BaseException, str, None]) -> str:
    """
    Herhangi bir hata metnini UI için sadeleştir.
    Kernel mismatch ise Türkçe rehber; değilse str(exc).
    """
    if exc_or_text is None:
        return ""
    if isinstance(exc_or_text, CudaKernelMismatchError):
        return str(exc_or_text)
    if is_cuda_kernel_mismatch(exc_or_text):
        return format_cuda_kernel_mismatch_message(original=exc_or_text)
    return str(exc_or_text)


def ensure_cuda_kernels_or_raise() -> None:
    """
    Model yüklemeden / warmup öncesi hızlı matmul kontrolü.
    CUDA yoksa sessizce döner (CPU mock yolu).
    """
    diag = diagnose_cuda()
    if not diag.get("cuda_available"):
        return
    result = verify_cuda_matmul()
    if result["ok"]:
        return
    err = result.get("error") or "CUDA matmul başarısız"
    if is_cuda_kernel_mismatch(err):
        raise CudaKernelMismatchError(
            format_cuda_kernel_mismatch_message(
                original=err, diagnosis=result.get("diagnosis") or diag
            )
        )
    raise RuntimeError(
        f"CUDA smoke test başarısız: {err}. "
        f"Teşhis: device={diag.get('device_name')} "
        f"cap={diag.get('capability')} torch={diag.get('torch_version')}"
    )
