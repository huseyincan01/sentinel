from src.pipeline.main_pipeline import SentinelPipeline

def test_dual_res():
    pipe = SentinelPipeline(vlm_size=336)
    assert pipe.vlm_size == 336
