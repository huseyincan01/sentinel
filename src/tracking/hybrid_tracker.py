"""
Hibrit Tracker: YOLOv8 + MOG2 (Basitleştirilmiş)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Sequence
import numpy as np

from src.tracking.mog2_detector import BlobDetection, MOG2Detector

YOLO_PREFIX = "yolo_"
MOG_PREFIX = "mog_"

@dataclass
class TrackedObject:
    track_id: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    source: str
    frame_idx: int
    with_reid: bool = False

@dataclass
class FrameTracks:
    frame_idx: int
    tracks: List[TrackedObject] = field(default_factory=list)
    yolo_detections: int = 0
    mog2_detections: int = 0

class SimpleIoUTracker:
    def __init__(self, match_thresh=0.3, max_age=30):
        self.match_thresh = match_thresh
        self.max_age = max_age
        self._next_id = 1
        self._tracks = {}

    def update(self, dets: np.ndarray) -> np.ndarray:
        if dets is None or len(dets) == 0:
            for tid in list(self._tracks.keys()):
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.max_age:
                    del self._tracks[tid]
            return np.zeros((0, 7), dtype=np.float32)

        for tid in self._tracks: self._tracks[tid]["age"] += 1

        matched_t, matched_d = set(), set()
        track_ids = list(self._tracks.keys())
        
        # Basit IoU eşleştirme
        for di, det in enumerate(dets):
            best_iou = 0
            best_ti = -1
            for ti, tid in enumerate(track_ids):
                if ti in matched_t: continue
                tb = self._tracks[tid]["bbox"]
                
                # IoU hesapla
                ix1, iy1 = max(tb[0], det[0]), max(tb[1], det[1])
                ix2, iy2 = min(tb[2], det[2]), min(tb[3], det[3])
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                if inter > 0:
                    area_t = max(0, tb[2] - tb[0]) * max(0, tb[3] - tb[1])
                    area_d = max(0, det[2] - det[0]) * max(0, det[3] - det[1])
                    iou = inter / (area_t + area_d - inter)
                    if iou > best_iou and iou >= self.match_thresh:
                        best_iou = iou
                        best_ti = ti
            
            conf = float(det[4]) if len(det) > 4 else 0.5
            cls = int(det[5]) if len(det) > 5 else 80
            if best_ti >= 0:
                matched_t.add(best_ti)
                matched_d.add(di)
                tid = track_ids[best_ti]
                self._tracks[tid].update({"bbox": det[:4], "age": 0, "conf": conf, "cls": cls})
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"bbox": det[:4], "age": 0, "conf": conf, "cls": cls}

        for tid in list(self._tracks.keys()):
            if self._tracks[tid]["age"] > self.max_age:
                del self._tracks[tid]

        res = [[*tr["bbox"], float(tid), tr["conf"], float(tr["cls"])] for tid, tr in self._tracks.items() if tr["age"] == 0]
        return np.array(res, dtype=np.float32) if res else np.zeros((0, 7), dtype=np.float32)

class HybridTracker:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
        classes: Optional[List[int]] = None,
        mog2: Optional[MOG2Detector] = None,
        yolo_model: Any = None,
        suppress_mog2_iou: float = 0.3,
        **kwargs
    ):
        self.conf = conf
        self.iou = iou
        self.device = device
        self.classes = classes
        self.suppress_mog2_iou = suppress_mog2_iou
        self._frame_idx = 0
        
        self.mog2 = mog2 or MOG2Detector()
        self._yolo = yolo_model
        self._model_path = model_path
        
        self._yolo_tracker = SimpleIoUTracker(max_age=15)
        self._mog_tracker = SimpleIoUTracker(max_age=10)
        self._names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 80: "mog2_blob"}

    @property
    def yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO
            self._yolo = YOLO(self._model_path)
            if hasattr(self._yolo, "names"):
                self._names.update({int(k): str(v) for k, v in self._yolo.names.items()})
        return self._yolo

    def reset(self) -> None:
        self._frame_idx = 0
        self.mog2.reset()
        self._yolo_tracker = SimpleIoUTracker(max_age=15)
        self._mog_tracker = SimpleIoUTracker(max_age=10)

    def _parse_tracks(self, raw: np.ndarray, prefix: str, source: str, frame_idx: int) -> List[TrackedObject]:
        if raw is None or len(raw) == 0: return []
        objects = []
        for row in raw:
            if len(row) < 5: continue
            x1, y1, x2, y2 = map(int, row[:4])
            track_id = f"{prefix}{int(row[4])}"
            conf = float(row[5]) if len(row) > 5 else 0.5
            cls_id = int(row[6]) if len(row) > 6 else 0
            cname = self._names.get(cls_id, "unknown") if source == "yolo" else "mog2_blob"
            objects.append(TrackedObject(track_id, (x1, y1, x2, y2), conf, cls_id, cname, source, frame_idx, False))
        return objects

    def process_frame(self, frame: np.ndarray, yolo_detections: Optional[np.ndarray] = None, run_mog2: bool = True) -> FrameTracks:
        idx = self._frame_idx
        self._frame_idx += 1
        
        # YOLO Pipeline
        yolo_dets = np.zeros((0, 6), dtype=np.float32)
        if yolo_detections is not None:
            yolo_dets = np.asarray(yolo_detections, dtype=np.float32).reshape(-1, 6) if len(yolo_detections) > 0 else yolo_dets
        else:
            if hasattr(self.yolo, "predict"):
                res = self.yolo.predict(frame, conf=self.conf, iou=self.iou, device=self.device, classes=self.classes, verbose=False)
                if res and res[0].boxes:
                    xyxy = res[0].boxes.xyxy.cpu().numpy()
                    conf = res[0].boxes.conf.cpu().numpy().reshape(-1, 1)
                    cls = res[0].boxes.cls.cpu().numpy().reshape(-1, 1)
                    yolo_dets = np.hstack([xyxy, conf, cls]).astype(np.float32)

        yolo_raw = self._yolo_tracker.update(yolo_dets)
        yolo_tracks = self._parse_tracks(yolo_raw, YOLO_PREFIX, "yolo", idx)

        # MOG2 Pipeline
        mog_tracks = []
        mog_det_count = 0
        if run_mog2:
            yolo_boxes = [t.bbox for t in yolo_tracks]
            for row in yolo_dets: yolo_boxes.append((int(row[0]), int(row[1]), int(row[2]), int(row[3])))
            
            blobs = self.mog2.detect(frame, exclude_boxes=yolo_boxes, iou_suppress=self.suppress_mog2_iou)
            mog_det_count = len(blobs)
            mog_arr = self.mog2.to_xyxy_conf(blobs)
            if len(mog_arr) > 0:
                cls_col = np.full((len(mog_arr), 1), 80.0, dtype=np.float32)
                mog_dets = np.hstack([mog_arr, cls_col])
            else:
                mog_dets = np.zeros((0, 6), dtype=np.float32)

            mog_raw = self._mog_tracker.update(mog_dets)
            mog_tracks = self._parse_tracks(mog_raw, MOG_PREFIX, "mog2", idx)

        return FrameTracks(idx, tracks=yolo_tracks + mog_tracks, yolo_detections=len(yolo_dets), mog2_detections=mog_det_count)

    def process_detections(self, yolo_detections: np.ndarray, mog2_detections: np.ndarray, frame: Optional[np.ndarray] = None, frame_idx: Optional[int] = None) -> FrameTracks:
        """Test mock desteği"""
        idx = self._frame_idx if frame_idx is None else frame_idx
        self._frame_idx = idx + 1
        
        y_dets = np.asarray(yolo_detections, dtype=np.float32).reshape(-1, 6) if len(yolo_detections) > 0 else np.zeros((0, 6), dtype=np.float32)
        
        m_dets = np.asarray(mog2_detections, dtype=np.float32)
        if len(m_dets) > 0:
            m_dets = m_dets.reshape(-1, m_dets.shape[-1])
            if m_dets.shape[1] == 5:
                m_dets = np.hstack([m_dets, np.full((len(m_dets), 1), 80.0, dtype=np.float32)])
        else:
            m_dets = np.zeros((0, 6), dtype=np.float32)

        y_raw = self._yolo_tracker.update(y_dets)
        m_raw = self._mog_tracker.update(m_dets)

        return FrameTracks(
            idx,
            tracks=self._parse_tracks(y_raw, YOLO_PREFIX, "yolo", idx) + self._parse_tracks(m_raw, MOG_PREFIX, "mog2", idx),
            yolo_detections=len(y_dets),
            mog2_detections=len(m_dets)
        )

    def get_reid_policy(self) -> Dict[str, bool]:
        return {"yolo": False, "mog2": False}
