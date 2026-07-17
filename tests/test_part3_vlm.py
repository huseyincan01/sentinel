from src.vlm.internvl_agent import InternVLAgent, make_mock_generator

def test_vlm_agent():
    agent = InternVLAgent(backend="mock", generator_fn=make_mock_generator())
    agent.load()
    assert agent.is_loaded
    res = agent.analyze_detail(trigger_info="test")
    assert hasattr(res, "summary")
