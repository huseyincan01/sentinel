import numpy as np
from src.tracking.hybrid_tracker import HybridTracker, YOLO_PREFIX, MOG_PREFIX

def test_hybrid_tracker():
    tracker = HybridTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = tracker.process_frame(frame, run_mog2=False)
    assert res.frame_idx == 0
    assert isinstance(res.tracks, list)
