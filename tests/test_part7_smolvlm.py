"""PART 7 — SmolVLM geliştirme backend + factory."""

from __future__ import annotations

from src.vlm.factory import (
    DEFAULT_INTERNVL_ID,
    DEFAULT_SMOLVLM_ID,
    create_vlm_agent,
)
from src.vlm.internvl_agent import InternVLAgent
from src.vlm.tools import ToolRegistry


class TestFactory:
    def test_mock_backend(self, tmp_path):
        tools = ToolRegistry(report_dir=tmp_path)
        agent = create_vlm_agent(backend="mock", tools=tools, auto_load=True)
        assert agent.backend == "mock"
        assert agent.is_loaded
        gate = agent.analyze_gate(
            frame=None, candidate_summary="entrance; stillness"
        )
        assert gate.need_high_res is True  # mock danger keywords

    def test_smolvlm_backend_config_without_download(self, tmp_path):
        """auto_load=False — model indirmeden yapılandırma."""
        tools = ToolRegistry(report_dir=tmp_path)
        agent = create_vlm_agent(
            backend="smolvlm", tools=tools, auto_load=False, device="cpu"
        )
        assert agent.backend == "smolvlm"
        assert agent.model_id == DEFAULT_SMOLVLM_ID
        assert agent.use_format_enforcer is False

    def test_internvl_backend_config(self, tmp_path):
        tools = ToolRegistry(report_dir=tmp_path)
        agent = create_vlm_agent(
            backend="internvl2", tools=tools, auto_load=False, device="cpu"
        )
        assert agent.backend == "internvl2"
        assert agent.model_id == DEFAULT_INTERNVL_ID

    def test_unknown_backend_raises(self, tmp_path):
        import pytest

        with pytest.raises(ValueError):
            create_vlm_agent(backend="gpt4v", tools=ToolRegistry(report_dir=tmp_path))


class TestInternVLAgentBackendField:
    def test_default_backend_smolvlm(self):
        a = InternVLAgent(backend="smolvlm", auto_execute_tools=False)
        # generator yok → model_id smol
        assert a.model_id == DEFAULT_SMOLVLM_ID or "SmolVLM" in a.model_id

    def test_mock_via_generators(self, tmp_path):
        from src.vlm.internvl_agent import make_mock_gate_generator, make_mock_generator

        a = InternVLAgent(
            backend="mock",
            tools=ToolRegistry(report_dir=tmp_path),
            generator_fn=make_mock_generator(),
            gate_generator_fn=make_mock_gate_generator(),
        )
        a.load()
        r = a.analyze_detail(frame_idx=1)
        assert r.summary
