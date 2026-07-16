"""
Triage — Kademe 1 aday sinyalleri + cooldown.

Adaylar VLM kırpma/bağlam sinyali üretir (tek VLM @336; high-res Detail yolu yok).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_metric

BBox = Tuple[int, int, int, int]
Point = Tuple[float, float]
Polygon = Sequence[Point]


class TriggerKind(str, Enum):
    SSIM = "ssim"
    STILLNESS = "stillness"
    DANGEROUS_MOTION = "dangerous_motion"
    ENTRANCE = "entrance"
    MOG2_MOTION = "mog2_motion"
    ROI = "roi"
    PERIODIC = "periodic"
    COLOR_FIRE = "color_fire"


class RiskLevel(str, Enum):
    LOW = "Düşük"
    MEDIUM = "Orta"
    HIGH = "Yüksek"
    CRITICAL = "Kritik"


TRIGGER_PRIORITY: Dict[TriggerKind, int] = {
    TriggerKind.SSIM: 1,
    TriggerKind.STILLNESS: 2,
    TriggerKind.DANGEROUS_MOTION: 3,
    TriggerKind.ENTRANCE: 4,
    TriggerKind.MOG2_MOTION: 5,
    TriggerKind.COLOR_FIRE: 6,
    TriggerKind.ROI: 7,
    TriggerKind.PERIODIC: 8,
}


@dataclass
class TriggerEvent:
    kind: TriggerKind
    priority: int
    timestamp: float
    frame_idx: int
    track_ids: List[str] = field(default_factory=list)
    ssim_value: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_bypass_capable(self) -> bool:
        return self.kind != TriggerKind.PERIODIC


@dataclass
class CandidateBundle:
    """Kademe 1 çıktısı — gate'e gidecek adaylar."""

    triggers: List[TriggerEvent] = field(default_factory=list)
    bypass_to_detail: bool = False
    bypass_reason: str = ""
    summary: str = ""

    @property
    def has_candidates(self) -> bool:
        return len(self.triggers) > 0


@dataclass
class TriageDecision:
    """
    Geriye uyumluluk / testler.

    should_call_vlm ≈ aday var ve (gate veya detail) eylemi düşünülmeli.
    Pipeline asıl orkestrasyonu yapar.
    """

    should_call_vlm: bool
    triggers: List[TriggerEvent] = field(default_factory=list)
    primary_trigger: Optional[TriggerEvent] = None
    cooldown_active: bool = False
    cooldown_bypassed: bool = False
    coalesced: bool = False
    reason: str = ""
    ssim_value: Optional[float] = None
    frame_idx: int = 0
    timestamp: float = 0.0
    bypass_to_detail: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_call_vlm": self.should_call_vlm,
            "triggers": [t.kind.value for t in self.triggers],
            "primary": self.primary_trigger.kind.value if self.primary_trigger else None,
            "cooldown_active": self.cooldown_active,
            "cooldown_bypassed": self.cooldown_bypassed,
            "coalesced": self.coalesced,
            "reason": self.reason,
            "ssim_value": self.ssim_value,
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "bypass_to_detail": self.bypass_to_detail,
        }


def _normalize_risk(risk: Union[str, RiskLevel, None]) -> RiskLevel:
    if risk is None:
        return RiskLevel.LOW
    if isinstance(risk, RiskLevel):
        return risk
    mapping = {
        "düşük": RiskLevel.LOW,
        "dusuk": RiskLevel.LOW,
        "low": RiskLevel.LOW,
        "orta": RiskLevel.MEDIUM,
        "medium": RiskLevel.MEDIUM,
        "yüksek": RiskLevel.HIGH,
        "yuksek": RiskLevel.HIGH,
        "high": RiskLevel.HIGH,
        "kritik": RiskLevel.CRITICAL,
        "critical": RiskLevel.CRITICAL,
    }
    return mapping.get(str(risk).strip().lower(), RiskLevel.LOW)


def bbox_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    if polygon is None or len(polygon) < 3:
        return False
    x, y = point
    pts = list(polygon)
    n = len(pts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def bbox_intersects_polygon(bbox: BBox, polygon: Polygon) -> bool:
    if polygon is None or len(polygon) < 3:
        return False
    x1, y1, x2, y2 = bbox
    if point_in_polygon(bbox_center(bbox), polygon):
        return True
    for c in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
        if point_in_polygon(c, polygon):
            return True
    for px, py in polygon:
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True
    return False


def _extract_track_info(track: Any) -> Tuple[str, BBox, str, int]:
    if isinstance(track, dict):
        tid = str(track.get("track_id", track.get("id", "unknown")))
        bb = track.get("bbox")
        if bb is None:
            raise ValueError("bbox yok")
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


def _is_person(cname: str, cid: int, tid: str) -> bool:
    cl = cname.lower()
    if "person" in cl or "insan" in cl:
        return True
    if cid == 0 and tid.startswith("yolo_"):
        return True
    return False


def _is_vehicle(cname: str, cid: int) -> bool:
    cl = cname.lower()
    if any(k in cl for k in ("car", "truck", "bus", "motorcycle", "forklift", "vehicle")):
        return True
    return cid in (2, 3, 5, 7)


class TriageEngine:
    def __init__(
        self,
        roi_polygon: Optional[Polygon] = None,
        ssim_threshold: float = 0.85,
        ssim_severe_threshold: float = 0.50,
        periodic_interval_s: float = 7.0,
        coalesce_window_ms: float = 900.0,
        cooldown_low_s: float = 6.0,
        cooldown_medium_s: float = 4.0,
        cooldown_high_s: float = 1.5,
        cooldown_critical_s: float = 1.0,
        cooldown_dedup_s: float = 45.0,
        ssim_resize: Tuple[int, int] = (320, 240),
        stillness_duration_s: float = 3.5,
        stillness_max_move_px: float = 12.0,
        motion_speed_px_s: float = 120.0,
        mog2_min_blobs: int = 1,
        entrance_enabled: bool = True,
        stillness_enabled: bool = True,
        mog2_trigger_enabled: bool = True,
        motion_enabled: bool = True,
        color_fire_enabled: bool = True,
        color_fire_threshold_px: int = 150,
        time_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.roi_polygon: Optional[List[Point]] = (
            list(roi_polygon) if roi_polygon is not None else None
        )
        self.ssim_threshold = float(ssim_threshold)
        self.ssim_severe_threshold = float(ssim_severe_threshold)
        self.periodic_interval_s = float(periodic_interval_s)
        self.coalesce_window_s = float(coalesce_window_ms) / 1000.0
        self.cooldown_low_s = float(cooldown_low_s)
        self.cooldown_medium_s = float(cooldown_medium_s)
        self.cooldown_high_s = float(cooldown_high_s)
        self.cooldown_critical_s = float(cooldown_critical_s)
        self.cooldown_dedup_s = float(cooldown_dedup_s)
        self.ssim_resize = ssim_resize
        self.stillness_duration_s = float(stillness_duration_s)
        self.stillness_max_move_px = float(stillness_max_move_px)
        self.motion_speed_px_s = float(motion_speed_px_s)
        self.mog2_min_blobs = int(mog2_min_blobs)
        self.entrance_enabled = entrance_enabled
        self.stillness_enabled = stillness_enabled
        self.mog2_trigger_enabled = mog2_trigger_enabled
        self.motion_enabled = motion_enabled
        self.color_fire_enabled = color_fire_enabled
        self.color_fire_threshold_px = int(color_fire_threshold_px)
        self._time_fn = time_fn or self._default_time

        self._prev_gray: Optional[np.ndarray] = None
        self._last_vlm_time: Optional[float] = None
        self._cooldown_until: float = 0.0
        self._current_cooldown_s: float = self.cooldown_low_s
        self._last_risk: RiskLevel = RiskLevel.LOW
        self._last_trigger_track_ids: List[str] = []
        self._roi_ids_seen: set = set()
        self._ids_in_roi_prev: set = set()
        self._low_risk_repeat: Dict[str, int] = {}
        self._global_seen_ids: set = set()
        self._center_history: Dict[str, List[Tuple[float, float, float]]] = {}
        self._last_center: Dict[str, Tuple[float, float, float]] = {}

        self._start_time: Optional[float] = None
        self._last_periodic_time: Optional[float] = None

        self.stats: Dict[str, int] = {
            "frames_evaluated": 0,
            "vlm_calls": 0,
            "gate_calls": 0,
            "detail_calls": 0,
            "roi_triggers": 0,
            "ssim_triggers": 0,
            "periodic_triggers": 0,
            "entrance_triggers": 0,
            "stillness_triggers": 0,
            "motion_triggers": 0,
            "mog2_triggers": 0,
            "color_fire_triggers": 0,
            "coalesced": 0,
            "cooldown_blocks": 0,
            "cooldown_bypasses": 0,
        }

    @staticmethod
    def _default_time() -> float:
        import time

        return time.monotonic()

    def now(self) -> float:
        return float(self._time_fn())

    def reset(self) -> None:
        self._prev_gray = None
        self._last_vlm_time = None
        self._cooldown_until = 0.0
        self._current_cooldown_s = self.cooldown_low_s
        self._last_risk = RiskLevel.LOW
        self._last_trigger_track_ids = []
        self._roi_ids_seen.clear()
        self._ids_in_roi_prev.clear()
        self._low_risk_repeat.clear()
        self._global_seen_ids.clear()
        self._center_history.clear()
        self._last_center.clear()
        self._start_time = None
        self._last_periodic_time = None
        for k in self.stats:
            self.stats[k] = 0

    def set_roi(self, polygon: Optional[Polygon]) -> None:
        self.roi_polygon = list(polygon) if polygon is not None else None
        self._roi_ids_seen.clear()
        self._ids_in_roi_prev.clear()

    def _frame_to_gray_small(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        w, h = self.ssim_resize
        return cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)

    # --- aday kontrolleri ---

    def check_ssim(self, frame, frame_idx, timestamp):
        gray = self._frame_to_gray_small(frame)
        if self._prev_gray is None:
            self._prev_gray = gray
            return None, None
        score = float(ssim_metric(self._prev_gray, gray, data_range=255))
        self._prev_gray = gray
        if score >= self.ssim_threshold:
            return None, score
        self.stats["ssim_triggers"] += 1
        severe = score < self.ssim_severe_threshold
        return (
            TriggerEvent(
                kind=TriggerKind.SSIM,
                priority=TRIGGER_PRIORITY[TriggerKind.SSIM],
                timestamp=timestamp,
                frame_idx=frame_idx,
                ssim_value=score,
                details={"severe": severe, "threshold": self.ssim_threshold},
            ),
            score,
        )

    def check_entrance(self, tracks, frame_idx, timestamp):
        if not self.entrance_enabled or not tracks:
            return None
        new_ids = []
        for tr in tracks:
            try:
                tid, _, cname, cid = _extract_track_info(tr)
            except Exception:
                continue
            if tid in self._global_seen_ids:
                continue
            if _is_person(cname, cid, tid) or _is_vehicle(cname, cid):
                new_ids.append(tid)
        for tr in tracks:
            try:
                tid, _, _, _ = _extract_track_info(tr)
                self._global_seen_ids.add(tid)
            except Exception:
                pass
        if not new_ids:
            return None
        self.stats["entrance_triggers"] += 1
        return TriggerEvent(
            kind=TriggerKind.ENTRANCE,
            priority=TRIGGER_PRIORITY[TriggerKind.ENTRANCE],
            timestamp=timestamp,
            frame_idx=frame_idx,
            track_ids=new_ids,
            details={"new_ids": new_ids},
        )

    def check_stillness(self, tracks, frame_idx, timestamp):
        if not self.stillness_enabled or not tracks:
            return None
        still_ids = []
        for tr in tracks:
            try:
                tid, bbox, cname, cid = _extract_track_info(tr)
            except Exception:
                continue
            if not _is_person(cname, cid, tid):
                continue
            cx, cy = bbox_center(bbox)
            hist = self._center_history.setdefault(tid, [])
            hist.append((timestamp, cx, cy))
            cutoff = timestamp - self.stillness_duration_s - 1.0
            self._center_history[tid] = [h for h in hist if h[0] >= cutoff]
            hist = self._center_history[tid]
            if len(hist) < 2:
                continue
            t0, x0, y0 = hist[0]
            if timestamp - t0 < self.stillness_duration_s:
                continue
            max_d = max(((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 for _, x, y in hist)
            if max_d <= self.stillness_max_move_px:
                still_ids.append(tid)
        if not still_ids:
            return None
        self.stats["stillness_triggers"] += 1
        return TriggerEvent(
            kind=TriggerKind.STILLNESS,
            priority=TRIGGER_PRIORITY[TriggerKind.STILLNESS],
            timestamp=timestamp,
            frame_idx=frame_idx,
            track_ids=still_ids,
            details={"duration_s": self.stillness_duration_s},
        )

    def check_dangerous_motion(self, tracks, frame_idx, timestamp):
        """Yüksek hız / ani bbox sıçraması."""
        if not self.motion_enabled or not tracks:
            return None
        fast_ids = []
        for tr in tracks:
            try:
                tid, bbox, cname, cid = _extract_track_info(tr)
            except Exception:
                continue
            if not (_is_person(cname, cid, tid) or _is_vehicle(cname, cid) or tid.startswith("yolo_")):
                continue
            cx, cy = bbox_center(bbox)
            prev = self._last_center.get(tid)
            self._last_center[tid] = (timestamp, cx, cy)
            if prev is None:
                continue
            t0, x0, y0 = prev
            dt = timestamp - t0
            if dt <= 1e-6:
                continue
            dist = ((cx - x0) ** 2 + (cy - y0) ** 2) ** 0.5
            speed = dist / dt
            if speed >= self.motion_speed_px_s:
                fast_ids.append(tid)
        if not fast_ids:
            return None
        self.stats["motion_triggers"] += 1
        return TriggerEvent(
            kind=TriggerKind.DANGEROUS_MOTION,
            priority=TRIGGER_PRIORITY[TriggerKind.DANGEROUS_MOTION],
            timestamp=timestamp,
            frame_idx=frame_idx,
            track_ids=fast_ids,
            details={"speed_threshold": self.motion_speed_px_s},
        )

    def check_mog2_motion(self, tracks, frame_idx, timestamp):
        if not self.mog2_trigger_enabled or not tracks:
            return None
        mog_ids = []
        for tr in tracks:
            try:
                tid, _, _, _ = _extract_track_info(tr)
            except Exception:
                continue
            source = (
                str(tr.get("source", ""))
                if isinstance(tr, dict)
                else str(getattr(tr, "source", "") or "")
            )
            if source == "mog2" or tid.startswith("mog_"):
                mog_ids.append(tid)
        if len(mog_ids) < self.mog2_min_blobs:
            return None
        self.stats["mog2_triggers"] += 1
        return TriggerEvent(
            kind=TriggerKind.MOG2_MOTION,
            priority=TRIGGER_PRIORITY[TriggerKind.MOG2_MOTION],
            timestamp=timestamp,
            frame_idx=frame_idx,
            track_ids=mog_ids,
            details={"mog_count": len(mog_ids)},
        )

    def check_roi(self, tracks, frame_idx, timestamp):
        if not self.roi_polygon or len(self.roi_polygon) < 3 or not tracks:
            return None
        inside = []
        for tr in tracks:
            try:
                tid, bbox, _, _ = _extract_track_info(tr)
            except Exception:
                continue
            if bbox_intersects_polygon(bbox, self.roi_polygon):
                inside.append(tid)
        if not inside:
            self._ids_in_roi_prev = set()
            return None
        self.stats["roi_triggers"] += 1
        new_ids = [i for i in inside if i not in self._roi_ids_seen]
        return TriggerEvent(
            kind=TriggerKind.ROI,
            priority=TRIGGER_PRIORITY[TriggerKind.ROI],
            timestamp=timestamp,
            frame_idx=frame_idx,
            track_ids=inside,
            details={"new_ids": new_ids},
        )

    def check_periodic(self, frame_idx, timestamp):
        if self._start_time is None:
            self._start_time = timestamp
            self._last_periodic_time = timestamp
            return None
        ref = self._last_periodic_time if self._last_periodic_time is not None else self._start_time
        if (timestamp - ref) < self.periodic_interval_s:
            return None
        self.stats["periodic_triggers"] += 1
        self._last_periodic_time = timestamp
        return TriggerEvent(
            kind=TriggerKind.PERIODIC,
            priority=TRIGGER_PRIORITY[TriggerKind.PERIODIC],
            timestamp=timestamp,
            frame_idx=frame_idx,
            details={"interval_s": self.periodic_interval_s},
        )

    def check_color_fire(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> Optional[TriggerEvent]:
        if not self.color_fire_enabled or frame is None or frame.size == 0:
            return None
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Alev tonları için HSV sınırları (kırmızı/turuncu/sarı aralığı)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([25, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([179, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            pixel_count = int(cv2.countNonZero(mask))
            if pixel_count >= self.color_fire_threshold_px:
                self.stats["color_fire_triggers"] += 1
                return TriggerEvent(
                    kind=TriggerKind.COLOR_FIRE,
                    priority=TRIGGER_PRIORITY[TriggerKind.COLOR_FIRE],
                    timestamp=timestamp,
                    frame_idx=frame_idx,
                    details={"fire_pixels": pixel_count, "threshold": self.color_fire_threshold_px},
                )
        except Exception:
            pass
        return None

    def collect_candidates(
        self,
        frame: np.ndarray,
        tracks: Optional[Sequence[Any]] = None,
        frame_idx: int = 0,
        timestamp: Optional[float] = None,
        skip_ssim: bool = False,
        skip_roi: bool = False,
        skip_periodic: bool = False,
        skip_entrance: bool = False,
        skip_stillness: bool = False,
        skip_mog2: bool = False,
        skip_motion: bool = False,
        skip_color_fire: bool = False,
    ) -> CandidateBundle:
        """Kademe 1: aday sinyalleri topla (gate/detail kararı yok)."""
        if frame is None or (hasattr(frame, "size") and frame.size == 0):
            raise ValueError("Boş kare")
        ts = self.now() if timestamp is None else float(timestamp)
        tracks = tracks or []
        raw: List[TriggerEvent] = []
        ssim_value = None

        if not skip_ssim:
            ev, ssim_value = self.check_ssim(frame, frame_idx, ts)
            if ev:
                raw.append(ev)
        if not skip_stillness:
            ev = self.check_stillness(tracks, frame_idx, ts)
            if ev:
                raw.append(ev)
        if not skip_motion:
            ev = self.check_dangerous_motion(tracks, frame_idx, ts)
            if ev:
                raw.append(ev)
        if not skip_entrance:
            ev = self.check_entrance(tracks, frame_idx, ts)
            if ev:
                raw.append(ev)
        else:
            for tr in tracks:
                try:
                    tid, _, _, _ = _extract_track_info(tr)
                    self._global_seen_ids.add(tid)
                except Exception:
                    pass
        if not skip_mog2:
            ev = self.check_mog2_motion(tracks, frame_idx, ts)
            if ev:
                raw.append(ev)
        if not skip_roi:
            ev = self.check_roi(tracks, frame_idx, ts)
            if ev:
                raw.append(ev)
        if not skip_periodic:
            ev = self.check_periodic(frame_idx, ts)
            if ev:
                raw.append(ev)
        if not skip_color_fire:
            ev = self.check_color_fire(frame, frame_idx, ts)
            if ev:
                raw.append(ev)

        raw.sort(key=lambda e: e.priority)
        bypass, bypass_reason = self.compute_bypass(raw)
        summary = "; ".join(
            f"{t.kind.value}:{t.details}" if t.details else t.kind.value for t in raw
        )
        return CandidateBundle(
            triggers=raw,
            bypass_to_detail=bypass,
            bypass_reason=bypass_reason,
            summary=summary or "aday yok",
        )

    def compute_bypass(self, triggers: List[TriggerEvent]) -> Tuple[bool, str]:
        """Gate atla → doğrudan detail."""
        for t in triggers:
            if t.kind == TriggerKind.PERIODIC:
                continue
            if t.kind == TriggerKind.SSIM and (
                t.details.get("severe")
                or (
                    t.ssim_value is not None
                    and t.ssim_value < self.ssim_severe_threshold
                )
            ):
                return True, "bypass: şiddetli SSIM"
            if t.kind == TriggerKind.STILLNESS:
                return True, "bypass: person hareketsizlik"
        return False, ""

    def is_in_cooldown(self, timestamp: Optional[float] = None) -> bool:
        ts = self.now() if timestamp is None else timestamp
        return ts < self._cooldown_until

    def _cooldown_remaining(self, timestamp: float) -> float:
        return max(0.0, self._cooldown_until - timestamp)

    def cooldown_seconds_for_risk(self, risk: Union[str, RiskLevel]) -> float:
        level = _normalize_risk(risk)
        if level == RiskLevel.CRITICAL:
            return self.cooldown_critical_s
        if level == RiskLevel.HIGH:
            return self.cooldown_high_s
        if level == RiskLevel.MEDIUM:
            return self.cooldown_medium_s
        return self.cooldown_low_s

    def notify_vlm_result(
        self,
        risk: Union[str, RiskLevel] = RiskLevel.LOW,
        track_ids: Optional[Sequence[str]] = None,
        timestamp: Optional[float] = None,
    ) -> float:
        """Detail VLM sonrası adaptif cooldown."""
        ts = self.now() if timestamp is None else float(timestamp)
        level = _normalize_risk(risk)
        ids = list(track_ids) if track_ids else list(self._last_trigger_track_ids)
        cd = self.cooldown_seconds_for_risk(level)
        if level in (RiskLevel.LOW, RiskLevel.MEDIUM) and ids:
            for tid in ids:
                self._low_risk_repeat[tid] = self._low_risk_repeat.get(tid, 0) + 1
                if self._low_risk_repeat[tid] >= 2:
                    cd = max(cd, self.cooldown_dedup_s)
        else:
            for tid in ids:
                self._low_risk_repeat.pop(tid, None)
        self._last_risk = level
        self._current_cooldown_s = cd
        self._cooldown_until = ts + cd
        self._last_vlm_time = ts
        self.stats["detail_calls"] += 1
        # vlm_calls escalate anında artar (evaluate); burada sadece detail
        return cd

    def mark_gate_called(self) -> None:
        self.stats["gate_calls"] += 1

    def evaluate(
        self,
        frame: np.ndarray,
        tracks: Optional[Sequence[Any]] = None,
        frame_idx: int = 0,
        timestamp: Optional[float] = None,
        skip_ssim: bool = False,
        skip_roi: bool = False,
        skip_periodic: bool = False,
        skip_entrance: bool = False,
        skip_stillness: bool = False,
        skip_mog2: bool = False,
        skip_motion: bool = False,
        skip_color_fire: bool = False,
    ) -> TriageDecision:
        """
        Test / geriye uyum: aday var mı + detail cooldown.

        should_call_vlm=True → pipeline gate veya detail düşünmeli.
        """
        ts = self.now() if timestamp is None else float(timestamp)
        self.stats["frames_evaluated"] += 1
        if self._start_time is None:
            self._start_time = ts

        bundle = self.collect_candidates(
            frame,
            tracks=tracks,
            frame_idx=frame_idx,
            timestamp=ts,
            skip_ssim=skip_ssim,
            skip_roi=skip_roi,
            skip_periodic=skip_periodic,
            skip_entrance=skip_entrance,
            skip_stillness=skip_stillness,
            skip_mog2=skip_mog2,
            skip_motion=skip_motion,
            skip_color_fire=skip_color_fire,
        )
        ssim_value = None
        for t in bundle.triggers:
            if t.kind == TriggerKind.SSIM:
                ssim_value = t.ssim_value

        if not bundle.has_candidates:
            return TriageDecision(
                should_call_vlm=False,
                reason="tetikleyici yok",
                ssim_value=ssim_value,
                frame_idx=frame_idx,
                timestamp=ts,
            )

        primary = bundle.triggers[0]
        in_cd = self.is_in_cooldown(ts)

        # Bypass detail cooldown kırar
        if in_cd and not bundle.bypass_to_detail:
            # Yeni entrance/ROI hâlâ testlerde bypass gibi davranabilir
            extra_bypass = False
            for t in bundle.triggers:
                if t.kind == TriggerKind.ENTRANCE and t.track_ids:
                    extra_bypass = True
                    break
                if t.kind == TriggerKind.ROI:
                    new_ids = t.details.get("new_ids") or []
                    if new_ids:
                        extra_bypass = True
                        break
            if not extra_bypass:
                self.stats["cooldown_blocks"] += 1
                return TriageDecision(
                    should_call_vlm=False,
                    triggers=bundle.triggers,
                    primary_trigger=primary,
                    cooldown_active=True,
                    reason=f"cooldown aktif ({self._cooldown_remaining(ts):.2f}s)",
                    ssim_value=ssim_value,
                    frame_idx=frame_idx,
                    timestamp=ts,
                )
            self.stats["cooldown_bypasses"] += 1
            self._cooldown_until = ts
            reason = "bypass cooldown"
            bypassed = True
        elif in_cd and bundle.bypass_to_detail:
            self.stats["cooldown_bypasses"] += 1
            self._cooldown_until = ts
            reason = bundle.bypass_reason or "bypass"
            bypassed = True
        else:
            reason = f"tetik: {primary.kind.value}"
            bypassed = False

        # Coalesce: yeni entrance bile pencere içinde tek eyleme birleşir
        # (bypass_to_detail / cooldown bypass ayrı; pure coalesce öncelikli)
        if (
            self._last_vlm_time is not None
            and (ts - self._last_vlm_time) < self.coalesce_window_s
            and not bundle.bypass_to_detail
            and not (bypassed and primary.kind in (TriggerKind.SSIM, TriggerKind.STILLNESS))
        ):
            # Normal aday tekrarı / yeni entrance → coalescing
            if not bypassed or primary.kind in (
                TriggerKind.ENTRANCE,
                TriggerKind.ROI,
                TriggerKind.MOG2_MOTION,
                TriggerKind.DANGEROUS_MOTION,
                TriggerKind.PERIODIC,
            ):
                self.stats["coalesced"] += 1
                return TriageDecision(
                    should_call_vlm=False,
                    triggers=bundle.triggers,
                    primary_trigger=primary,
                    coalesced=True,
                    reason="coalescing",
                    ssim_value=ssim_value,
                    frame_idx=frame_idx,
                    timestamp=ts,
                )

        # Aday onaylandı — gate/detail pipeline'da
        # Test uyumu: escalate anını işaretle (coalesce / vlm_calls sayacı)
        self._last_vlm_time = ts
        self.stats["vlm_calls"] += 1
        self._last_trigger_track_ids = []
        for t in bundle.triggers:
            self._last_trigger_track_ids.extend(t.track_ids)
            if t.kind == TriggerKind.ROI:
                self._roi_ids_seen.update(t.track_ids)
                self._ids_in_roi_prev = set(t.track_ids)

        return TriageDecision(
            should_call_vlm=True,
            triggers=bundle.triggers,
            primary_trigger=primary,
            cooldown_bypassed=bypassed,
            reason=reason,
            ssim_value=ssim_value,
            frame_idx=frame_idx,
            timestamp=ts,
            bypass_to_detail=bundle.bypass_to_detail,
        )

    def savings_ratio(self, baseline_fps: float = 1.0, duration_s: Optional[float] = None) -> float:
        frames = max(self.stats["frames_evaluated"], 1)
        calls = self.stats.get("detail_calls", 0) or self.stats["vlm_calls"]
        baseline = baseline_fps * duration_s if duration_s and duration_s > 0 else frames
        return max(0.0, 1.0 - (calls / max(baseline, 1)))
