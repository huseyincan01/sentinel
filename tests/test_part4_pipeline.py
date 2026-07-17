import numpy as np
from src.pipeline.main_pipeline import SentinelPipeline

def test_pipeline():
    pipe = SentinelPipeline()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    res = pipe.process_frame(frame)
    assert res.frame_idx == 0
