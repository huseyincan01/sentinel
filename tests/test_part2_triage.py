import numpy as np
from src.decision.triage_engine import TriageEngine

def test_triage_engine():
    engine = TriageEngine()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    res = engine.evaluate(frame, tracks=[{"id": "yolo_1", "bbox": [0,0,10,10]}])
    assert res.should_call_vlm is True
    assert res.has_motion is True
