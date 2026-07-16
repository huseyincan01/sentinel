"""CUDA kernel mismatch teşhis ve UI mesaj testleri (GPU gerektirmez)."""

from __future__ import annotations

import pytest

from src.vlm.cuda_compat import (
    CudaKernelMismatchError,
    diagnose_cuda,
    format_cuda_kernel_mismatch_message,
    humanize_cuda_error,
    is_cuda_kernel_mismatch,
    raise_if_cuda_kernel_mismatch,
)


class TestIsCudaKernelMismatch:
    def test_classic_message(self):
        msg = (
            "AcceleratorError: CUDA error: no kernel image is available "
            "for execution on the device"
        )
        assert is_cuda_kernel_mismatch(msg) is True

    def test_cudaErrorNoKernelImage(self):
        assert is_cuda_kernel_mismatch("Search for `cudaErrorNoKernelImage`") is True

    def test_exception_instance(self):
        exc = RuntimeError(
            "CUDA error: no kernel image is available for execution on the device"
        )
        assert is_cuda_kernel_mismatch(exc) is True

    def test_custom_type(self):
        err = CudaKernelMismatchError("uyumsuz")
        assert is_cuda_kernel_mismatch(err) is True

    def test_unrelated_oom(self):
        assert is_cuda_kernel_mismatch("CUDA out of memory") is False

    def test_none_and_empty(self):
        assert is_cuda_kernel_mismatch(None) is False
        assert is_cuda_kernel_mismatch("") is False

    def test_chained_cause(self):
        root = RuntimeError(
            "no kernel image is available for execution on the device"
        )
        wrap = RuntimeError("generate failed")
        wrap.__cause__ = root
        assert is_cuda_kernel_mismatch(wrap) is True


class TestHumanizeAndRaise:
    def test_humanize_contains_turkish_guidance(self):
        raw = "CUDA error: no kernel image is available for execution on the device"
        text = humanize_cuda_error(raw)
        assert "CUDA mimari uyumsuzluğu" in text
        assert "torch" in text.lower()
        assert "kaggle" in text.lower() or "Kaggle" in text

    def test_humanize_passthrough_other(self):
        assert humanize_cuda_error("basit hata") == "basit hata"

    def test_raise_if_mismatch(self):
        with pytest.raises(CudaKernelMismatchError) as ei:
            raise_if_cuda_kernel_mismatch(
                RuntimeError(
                    "no kernel image is available for execution on the device"
                )
            )
        assert "CUDA mimari" in str(ei.value)

    def test_raise_if_not_mismatch_noop(self):
        raise_if_cuda_kernel_mismatch(ValueError("json bozuk"))  # no raise

    def test_format_includes_optional_diag(self):
        diag = {
            "device_name": "Tesla T4",
            "capability": "sm_75",
            "torch_version": "2.4.0+cu121",
            "torch_cuda_built": "12.1",
            "arch_list": ["sm_70", "sm_75", "sm_80"],
            "capability_supported": True,
        }
        msg = format_cuda_kernel_mismatch_message(
            original="no kernel image",
            diagnosis=diag,
        )
        assert "Tesla T4" in msg
        assert "sm_75" in msg


class TestDiagnoseSafe:
    def test_diagnose_returns_dict(self):
        d = diagnose_cuda()
        assert isinstance(d, dict)
        assert "cuda_available" in d
        assert "torch_version" in d


class TestAgentGenerateWrap:
    """Mock generate yolunda kernel hatası dönüşümü (model yok)."""

    def test_generate_wraps_kernel_error(self, monkeypatch):
        from src.vlm.internvl_agent import InternVLAgent

        agent = InternVLAgent(backend="smolvlm", generator_fn=None)
        agent._loaded = True
        agent.model = object()  # truthy
        agent.processor = object()
        agent.backend = "smolvlm"

        def boom(*_a, **_k):
            raise RuntimeError(
                "CUDA error: no kernel image is available for execution on the device"
            )

        monkeypatch.setattr(agent, "_generate_smolvlm", boom)

        with pytest.raises(CudaKernelMismatchError):
            agent._generate_text("test", image=None)
