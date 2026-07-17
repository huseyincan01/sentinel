"""
Triage — Aday tespiti (Basitleştirilmiş Sürüm)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

BBox = Tuple[int, int, int, int]
Point = Tuple[float, float]
Polygon = Sequence[Point]

class TriggerKind(str, Enum):
    MOTION = "motion"
    ROI = "roi"
    PERIODIC = "periodic"

@dataclass
class TriggerEvent:
    kind: TriggerKind
    timestamp: float
    frame_idx: int
    track_ids: List[str] = field(default_factory=list)

@dataclass
class TriageDecision:
    should_call_vlm: bool
    triggers: List[TriggerEvent] = field(default_factory=list)
    has_motion: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_call_vlm": self.should_call_vlm,
            "has_motion": self.has_motion,
            "triggers": [t.kind.value for t in self.triggers]
        }

def bbox_intersects_polygon(bbox: BBox, polygon: Polygon) -> bool:
    if not polygon or len(polygon) < 3: return False
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    
    # Ray-casting alg
    def in_poly(x, y):
        pts = list(polygon)
        n = len(pts)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        return inside
    
    if in_poly(cx, cy): return True
    for c in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
        if in_poly(c[0], c[1]): return True
    return False

def _extract_track_info(track: Any) -> Tuple[str, BBox, str, int]:
    if isinstance(track, dict):
        tid = str(track.get("track_id", track.get("id", "unknown")))
        bb = track.get("bbox")
        bbox = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
        cname = str(track.get("class_name", track.get("label", "")))
        cid = int(track.get("class_id", 0))
        return tid, bbox, cname, cid
    tid = str(getattr(track, "track_id"))
    bb = getattr(track, "bbox")
    bbox = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
    cname = str(getattr(track, "class_name", "") or "")
    cid = int(getattr(track, "class_id", 0) or 0)
    return tid, bbox, cname, cid

class TriageEngine:
    """Minimal triage engine. Geriye uyumluluk için notify_vlm_result ve evaluate fonksiyonları."""
    def __init__(
        self,
        roi_polygon: Optional[Polygon] = None,
        periodic_interval_s: float = 7.0,
        time_fn: Optional[Callable[[], float]] = None,
        **kwargs  # Eski parametreleri yok saymak için
    ):
        self.roi_polygon = list(roi_polygon) if roi_polygon else None
        self.periodic_interval_s = float(periodic_interval_s)
        self._time_fn = time_fn or __import__('time').monotonic
        self._last_periodic_time: Optional[float] = None
        self.stats = {
            "frames_evaluated": 0, "roi_triggers": 0, "periodic_triggers": 0,
            "motion_triggers": 0, "vlm_calls": 0, "gate_calls": 0, "detail_calls": 0
        }

    def reset(self) -> None:
        self._last_periodic_time = None
        for k in self.stats: self.stats[k] = 0

    def notify_vlm_result(self, risk: Any = None, track_ids: Any = None, timestamp: Any = None) -> float:
        self.stats["detail_calls"] += 1
        return 0.0

    def mark_gate_called(self) -> None:
        self.stats["gate_calls"] += 1

    def evaluate(
        self,
        frame: np.ndarray,
        tracks: Optional[Sequence[Any]] = None,
        frame_idx: int = 0,
        timestamp: Optional[float] = None,
        **kwargs
    ) -> TriageDecision:
        ts = timestamp if timestamp is not None else self._time_fn()
        self.stats["frames_evaluated"] += 1
        
        if self._last_periodic_time is None:
            self._last_periodic_time = ts
            
        tracks = tracks or []
        triggers = []
        has_motion = False
        track_ids = []

        # Hareket kontrolü (MOG2 veya YOLO trackleri)
        for tr in tracks:
            try:
                tid, bbox, cname, cid = _extract_track_info(tr)
                track_ids.append(tid)
                source = str(tr.get("source", "")) if isinstance(tr, dict) else str(getattr(tr, "source", "") or "")
                
                # YOLO nesnesi veya MOG2 hareketi
                if source == "mog2" or tid.startswith("mog_") or tid.startswith("yolo_"):
                    has_motion = True
                    
                # ROI Kontrolü
                if self.roi_polygon and bbox_intersects_polygon(bbox, self.roi_polygon):
                    triggers.append(TriggerEvent(TriggerKind.ROI, ts, frame_idx, [tid]))
                    self.stats["roi_triggers"] += 1
            except Exception:
                continue

        if has_motion:
            triggers.append(TriggerEvent(TriggerKind.MOTION, ts, frame_idx, track_ids))
            self.stats["motion_triggers"] += 1

        # Periyodik tetikleyici
        if (ts - self._last_periodic_time) >= self.periodic_interval_s:
            triggers.append(TriggerEvent(TriggerKind.PERIODIC, ts, frame_idx, []))
            self._last_periodic_time = ts
            self.stats["periodic_triggers"] += 1

        # Herhangi bir hareket varsa veya ROI/Periyodik tetiklendiğinde VLM çalışmalı (Crop ile veya değil)
        should_call = len(triggers) > 0
        
        return TriageDecision(
            should_call_vlm=should_call,
            triggers=triggers,
            has_motion=has_motion
        )
