import numpy as np

from src.pipeline.main_pipeline import build_demo_pipeline
from src.vlm.internvl_agent import InternVLAgent, make_mock_generator
from src.vlm.tools import ToolRegistry


def test_pipeline_honors_requested_mock_backend():
    pipeline = build_demo_pipeline(mock_vlm=True)
    assert pipeline.agent.backend == "mock"
    assert pipeline.tracker.device == "cpu"


def test_mock_pipeline_produces_schema_result_at_336():
    pipeline = build_demo_pipeline(mock_vlm=True, vlm_size=336, vlm_period_s=0)
    result = pipeline.process_frame(np.zeros((120, 240, 3), dtype=np.uint8))
    assert result.vlm_input_shape == (336, 336, 3)
    assert pipeline.get_last_summary()
    assert pipeline.get_last_score() == 0.1


def test_prepare_for_streaming_loads_agent_and_tracker_before_video():
    class FakeAgent:
        is_loaded = False

        def load(self):
            self.is_loaded = True

    class FakeTracker:
        yolo = object()

    from src.pipeline.main_pipeline import SentinelPipeline
    pipeline = SentinelPipeline(agent=FakeAgent(), tracker=FakeTracker())
    pipeline.prepare_for_streaming()
    assert pipeline.agent.is_loaded


def test_agent_parses_json_and_executes_declared_tools(tmp_path):
    tools = ToolRegistry(report_dir=tmp_path)
    agent = InternVLAgent(
        backend="mock",
        tools=tools,
        generator_fn=make_mock_generator({
            "summary": "Yaralanma riski var.",
            "events": [{"time": "00:01", "event": "Kişi yerde", "severity": "Yüksek"}],
            "risk": "Yüksek",
            "risk_score": 0.9,
            "actions": ["Alanı kapat"],
            "tools_called": ["lock_area"],
        }),
    )
    result = agent.analyze_detail(frame_idx=7, execute_tools=True)
    assert result.frame_analyzed == 7
    assert result.risk == "Yüksek"
    assert [item.tool for item in agent.last_tool_results] == ["lock_area"]
