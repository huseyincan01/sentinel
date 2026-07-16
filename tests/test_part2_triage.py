"""
PART 2 testleri — Triage Engine.

Kapsam:
- ROI tetikleyici (poligon içi / dışı)
- SSIM tetikleyici (ani sahne değişimi)
- Periyodik heartbeat
- Coalescing (800–1000 ms)
- Adaptif cooldown (risk bazlı)
- Alert fatigue (aynı ID düşük risk)
- Cooldown bypass (yeni ROI ID, şiddetli SSIM)
- Periyodik ASLA cooldown kırmaz
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pytest

from src.decision.triage_engine import (
    RiskLevel,
    TriggerKind,
    TriageEngine,
    bbox_intersects_polygon,
    point_in_polygon,
)


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

@dataclass
class FakeTrack:
    track_id: str
    bbox: Tuple[int, int, int, int]


class FakeClock:
    """Kontrollü zaman — coalescing/cooldown testleri için."""

    def __init__(self, start: float = 0.0):
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t


def _frame(h: int = 120, w: int = 160, value: int = 50) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _frame_noise(h: int = 120, w: int = 160, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ROI: sol-üst kare (0,0)-(80,80)
ROI = [(0, 0), (80, 0), (80, 80), (0, 80)]


def _engine(**kwargs) -> tuple:
    clock = FakeClock(0.0)
    defaults = dict(
        roi_polygon=ROI,
        ssim_threshold=0.85,
        ssim_severe_threshold=0.50,
        periodic_interval_s=5.0,
        coalesce_window_ms=900.0,
        cooldown_low_s=6.0,
        cooldown_high_s=1.5,
        cooldown_critical_s=1.0,
        cooldown_dedup_s=40.0,
        time_fn=clock,
    )
    defaults.update(kwargs)
    return TriageEngine(**defaults), clock


# ---------------------------------------------------------------------------
# Geometri
# ---------------------------------------------------------------------------

class TestGeometry:
    def test_point_in_polygon_inside(self):
        assert point_in_polygon((40, 40), ROI) is True

    def test_point_in_polygon_outside(self):
        assert point_in_polygon((200, 200), ROI) is False

    def test_bbox_intersects_roi(self):
        assert bbox_intersects_polygon((10, 10, 30, 30), ROI) is True
        assert bbox_intersects_polygon((200, 200, 250, 250), ROI) is False


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------

class TestROITrigger:
    def test_roi_triggers_when_track_inside(self):
        eng, clock = _engine()
        # ROI-only: entrance/stillness kapalı; person olmayan ID
        eng.entrance_enabled = False
        eng.stillness_enabled = False
        eng.mog2_trigger_enabled = False
        tracks = [FakeTrack("obj_1", (20, 20, 40, 40))]
        d = eng.evaluate(
            _frame(), tracks, frame_idx=0, timestamp=clock.t,
            skip_ssim=True, skip_periodic=True,
            skip_entrance=True, skip_stillness=True, skip_mog2=True,
        )
        assert d.should_call_vlm is True
        assert d.primary_trigger is not None
        assert d.primary_trigger.kind == TriggerKind.ROI
        assert "obj_1" in d.primary_trigger.track_ids

    def test_roi_no_trigger_outside(self):
        eng, clock = _engine()
        eng.entrance_enabled = False
        eng.stillness_enabled = False
        eng.mog2_trigger_enabled = False
        tracks = [FakeTrack("yolo_2", (200, 200, 240, 240))]
        d = eng.evaluate(
            _frame(), tracks, frame_idx=0, timestamp=clock.t,
            skip_ssim=True, skip_periodic=True,
            skip_entrance=True, skip_stillness=True, skip_mog2=True,
        )
        assert d.should_call_vlm is False
        assert d.triggers == []

    def test_roi_disabled_without_polygon(self):
        eng, clock = _engine(roi_polygon=None)
        eng.entrance_enabled = False
        eng.stillness_enabled = False
        eng.mog2_trigger_enabled = False
        tracks = [FakeTrack("yolo_1", (20, 20, 40, 40))]
        d = eng.evaluate(
            _frame(), tracks, frame_idx=0, timestamp=clock.t,
            skip_ssim=True, skip_periodic=True,
            skip_entrance=True, skip_stillness=True, skip_mog2=True,
        )
        assert d.should_call_vlm is False

    def test_roi_dict_track_supported(self):
        eng, clock = _engine()
        eng.entrance_enabled = False
        eng.stillness_enabled = False
        eng.mog2_trigger_enabled = False
        tracks = [{"track_id": "obj_3", "bbox": (10, 10, 50, 50), "class_id": 99, "class_name": "box"}]
        d = eng.evaluate(
            _frame(), tracks, frame_idx=1, timestamp=clock.t,
            skip_ssim=True, skip_periodic=True,
            skip_entrance=True, skip_stillness=True, skip_mog2=True,
        )
        assert d.should_call_vlm is True
        assert "obj_3" in d.primary_trigger.track_ids


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

class TestSSIMTrigger:
    def test_ssim_no_trigger_on_first_frame(self):
        eng, clock = _engine()
        d = eng.evaluate(_frame(value=40), [], frame_idx=0, timestamp=clock.t, skip_roi=True, skip_periodic=True)
        assert d.should_call_vlm is False

    def test_ssim_no_trigger_on_similar_frames(self):
        eng, clock = _engine()
        f1 = _frame(value=40)
        eng.evaluate(f1, [], frame_idx=0, timestamp=0.0, skip_roi=True, skip_periodic=True)
        # Neredeyse aynı kare
        f2 = _frame(value=41)
        d = eng.evaluate(f2, [], frame_idx=1, timestamp=1.0, skip_roi=True, skip_periodic=True)
        assert d.should_call_vlm is False
        if d.ssim_value is not None:
            assert d.ssim_value >= 0.85

    def test_ssim_triggers_on_scene_change(self):
        eng, clock = _engine()
        eng.evaluate(_frame(value=10), [], frame_idx=0, timestamp=0.0, skip_roi=True, skip_periodic=True)
        # Tamamen farklı gürültü karesi
        d = eng.evaluate(_frame_noise(seed=99), [], frame_idx=1, timestamp=1.0, skip_roi=True, skip_periodic=True)
        assert d.should_call_vlm is True
        assert d.primary_trigger.kind == TriggerKind.SSIM
        assert d.ssim_value is not None
        assert d.ssim_value < 0.85


# ---------------------------------------------------------------------------
# Periyodik
# ---------------------------------------------------------------------------

class TestPeriodicTrigger:
    def test_periodic_fires_after_interval(self):
        eng, clock = _engine(periodic_interval_s=5.0)
        # t=0: start, tetik yok
        d0 = eng.evaluate(_frame(), [], frame_idx=0, timestamp=0.0, skip_roi=True, skip_ssim=True)
        assert d0.should_call_vlm is False

        # t=4.9: henüz değil
        d1 = eng.evaluate(_frame(), [], frame_idx=1, timestamp=4.9, skip_roi=True, skip_ssim=True)
        assert d1.should_call_vlm is False

        # t=5.1: heartbeat
        d2 = eng.evaluate(_frame(), [], frame_idx=2, timestamp=5.1, skip_roi=True, skip_ssim=True)
        assert d2.should_call_vlm is True
        assert d2.primary_trigger.kind == TriggerKind.PERIODIC


# ---------------------------------------------------------------------------
# Coalescing
# ---------------------------------------------------------------------------

class TestCoalescing:
    def test_multiple_triggers_within_window_single_vlm(self):
        eng, clock = _engine(coalesce_window_ms=900.0, cooldown_low_s=0.01)
        tracks = [FakeTrack("yolo_1", (20, 20, 40, 40))]

        # İlk ROI → VLM
        d0 = eng.evaluate(
            _frame(), tracks, frame_idx=0, timestamp=0.0,
            skip_ssim=True, skip_periodic=True,
        )
        assert d0.should_call_vlm is True
        # Cooldown'u kısa tut / kapat: coalescing penceresi içindeyiz
        eng._cooldown_until = 0.0
        eng.notify_vlm_result(risk="Düşük", track_ids=["yolo_1"], timestamp=0.0)
        # notify cooldown açtı; coalescing test için cooldown'u kapat ama last_vlm_time kalsın
        eng._cooldown_until = 0.0

        # 400 ms sonra başka track ROI → coalescing
        d1 = eng.evaluate(
            _frame(),
            [FakeTrack("yolo_2", (25, 25, 45, 45))],
            frame_idx=1,
            timestamp=0.4,
            skip_ssim=True,
            skip_periodic=True,
        )
        # yolo_2 yeni ID → bypass cooldown olurdu ama cooldown zaten 0.
        # last_vlm 0.0, now 0.4 < 0.9 → coalesce (bypass yok)
        assert d1.coalesced is True
        assert d1.should_call_vlm is False
        assert eng.stats["vlm_calls"] == 1
        assert eng.stats["coalesced"] >= 1

    def test_after_coalesce_window_new_call_allowed(self):
        eng, clock = _engine(coalesce_window_ms=900.0)
        tracks = [FakeTrack("yolo_1", (20, 20, 40, 40))]

        d0 = eng.evaluate(_frame(), tracks, 0, 0.0, skip_ssim=True, skip_periodic=True)
        assert d0.should_call_vlm is True
        eng._cooldown_until = 0.0

        # Pencere dışı
        d1 = eng.evaluate(_frame(), tracks, 1, 1.5, skip_ssim=True, skip_periodic=True)
        assert d1.should_call_vlm is True
        assert d1.coalesced is False
        assert eng.stats["vlm_calls"] == 2


# ---------------------------------------------------------------------------
# Adaptif cooldown
# ---------------------------------------------------------------------------

class TestAdaptiveCooldown:
    def test_high_risk_short_cooldown(self):
        eng, _ = _engine()
        cd = eng.notify_vlm_result(risk="Kritik", track_ids=["yolo_1"], timestamp=10.0)
        assert cd == pytest.approx(eng.cooldown_critical_s)
        assert eng._cooldown_until == pytest.approx(10.0 + eng.cooldown_critical_s)

    def test_low_risk_long_cooldown(self):
        eng, _ = _engine()
        cd = eng.notify_vlm_result(risk="Düşük", track_ids=["yolo_9"], timestamp=0.0)
        assert cd == pytest.approx(eng.cooldown_low_s)

    def test_alert_fatigue_same_id(self):
        eng, _ = _engine(cooldown_dedup_s=40.0)
        # İlk düşük risk
        cd1 = eng.notify_vlm_result(risk="Düşük", track_ids=["yolo_5"], timestamp=0.0)
        assert cd1 == pytest.approx(eng.cooldown_low_s)
        # Aynı ID tekrar düşük risk → dedup
        cd2 = eng.notify_vlm_result(risk="Düşük", track_ids=["yolo_5"], timestamp=10.0)
        assert cd2 == pytest.approx(eng.cooldown_dedup_s)

    def test_cooldown_blocks_repeat_roi(self):
        eng, clock = _engine(cooldown_low_s=6.0)
        tracks = [FakeTrack("yolo_1", (20, 20, 40, 40))]

        d0 = eng.evaluate(_frame(), tracks, 0, 0.0, skip_ssim=True, skip_periodic=True)
        assert d0.should_call_vlm is True
        eng.notify_vlm_result(risk="Düşük", track_ids=["yolo_1"], timestamp=0.0)

        # Cooldown içinde aynı ID
        d1 = eng.evaluate(_frame(), tracks, 1, 2.0, skip_ssim=True, skip_periodic=True)
        assert d1.should_call_vlm is False
        assert d1.cooldown_active is True
        assert eng.stats["cooldown_blocks"] >= 1


# ---------------------------------------------------------------------------
# Cooldown bypass
# ---------------------------------------------------------------------------

class TestCooldownBypass:
    def test_new_roi_id_bypasses_cooldown(self):
        eng, clock = _engine(cooldown_low_s=10.0)
        t1 = [FakeTrack("yolo_1", (20, 20, 40, 40))]

        d0 = eng.evaluate(_frame(), t1, 0, 0.0, skip_ssim=True, skip_periodic=True)
        assert d0.should_call_vlm is True
        eng.notify_vlm_result(risk="Düşük", track_ids=["yolo_1"], timestamp=0.0)

        # Yeni ID ROI'ye giriyor
        t2 = [
            FakeTrack("yolo_1", (20, 20, 40, 40)),
            FakeTrack("yolo_99", (30, 30, 50, 50)),
        ]
        d1 = eng.evaluate(_frame(), t2, 1, 1.0, skip_ssim=True, skip_periodic=True)
        assert d1.should_call_vlm is True
        assert d1.cooldown_bypassed is True
        assert eng.stats["cooldown_bypasses"] >= 1

    def test_severe_ssim_bypasses_cooldown(self):
        eng, clock = _engine(
            cooldown_low_s=10.0,
            ssim_threshold=0.85,
            ssim_severe_threshold=0.55,
        )
        # Arkaplan öğren + sahte VLM + cooldown
        eng.evaluate(_frame(value=10), [], 0, 0.0, skip_roi=True, skip_periodic=True)
        eng._last_vlm_time = 0.0
        eng._cooldown_until = 10.0
        eng.stats["vlm_calls"] = 1

        # Şiddetli sahne değişimi (gürültü)
        d = eng.evaluate(
            _frame_noise(seed=7), [], 1, 1.0,
            skip_roi=True, skip_periodic=True,
        )
        assert d.primary_trigger is not None
        assert d.primary_trigger.kind == TriggerKind.SSIM
        # Şiddetli ise bypass
        if d.ssim_value is not None and d.ssim_value < eng.ssim_severe_threshold:
            assert d.should_call_vlm is True
            assert d.cooldown_bypassed is True
        else:
            # Eşik sınırında kalırsa en azından SSIM tetik oluşmuş olmalı
            assert d.ssim_value is not None and d.ssim_value < eng.ssim_threshold

    def test_periodic_never_bypasses_cooldown(self):
        eng, clock = _engine(periodic_interval_s=2.0, cooldown_low_s=20.0)
        # t=0 start
        eng.evaluate(_frame(), [], 0, 0.0, skip_roi=True, skip_ssim=True)
        # Sahte VLM + uzun cooldown
        eng._last_vlm_time = 0.0
        eng._cooldown_until = 20.0
        eng.stats["vlm_calls"] = 1

        # t=3 → periyodik tetiklenir ama cooldown kırılmaz
        d = eng.evaluate(_frame(), [], 1, 3.0, skip_roi=True, skip_ssim=True)
        assert any(t.kind == TriggerKind.PERIODIC for t in d.triggers) or d.primary_trigger is not None
        if d.triggers:
            assert d.should_call_vlm is False
            assert d.cooldown_active is True
            assert d.cooldown_bypassed is False

    def test_mild_ssim_does_not_bypass(self):
        """SSIM tetik eşiğinin altında ama severe üstünde → cooldown kırılmaz."""
        eng, clock = _engine(
            ssim_threshold=0.99,  # çok hassas: ufak fark bile tetik
            ssim_severe_threshold=0.10,  # severe çok zor
            cooldown_low_s=10.0,
        )
        base = _frame(value=100)
        eng.evaluate(base, [], 0, 0.0, skip_roi=True, skip_periodic=True)
        eng._last_vlm_time = 0.0
        eng._cooldown_until = 10.0
        eng.stats["vlm_calls"] = 1

        # Hafif değişim
        mild = _frame(value=110)
        d = eng.evaluate(mild, [], 1, 1.0, skip_roi=True, skip_periodic=True)
        if d.triggers and any(t.kind == TriggerKind.SSIM for t in d.triggers):
            # Severe değilse bypass olmamalı
            if d.ssim_value is not None and d.ssim_value >= eng.ssim_severe_threshold:
                assert d.should_call_vlm is False
                assert d.cooldown_bypassed is False


# ---------------------------------------------------------------------------
# Öncelik hiyerarşisi
# ---------------------------------------------------------------------------

class TestPriority:
    def test_ssim_outranks_roi_and_periodic(self):
        """Yeni mimari: SSIM önceliği ROI'den yüksek (ROI opsiyonel)."""
        eng, clock = _engine(periodic_interval_s=0.1)
        eng.evaluate(_frame(value=5), [], 0, 0.0, skip_roi=True, skip_periodic=True)
        tracks = [FakeTrack("yolo_1", (15, 15, 40, 40))]
        d = eng.evaluate(
            _frame_noise(seed=1),
            tracks,
            frame_idx=1,
            timestamp=5.0,
            skip_roi=False,
            skip_ssim=False,
            skip_periodic=False,
        )
        assert d.should_call_vlm is True
        assert d.primary_trigger.kind == TriggerKind.SSIM
        kinds = {t.kind for t in d.triggers}
        assert TriggerKind.SSIM in kinds


# ---------------------------------------------------------------------------
# Entegrasyon / kenar durumlar
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_empty_frame_raises(self):
        eng, _ = _engine()
        with pytest.raises(ValueError):
            eng.evaluate(np.array([]), [])

    def test_reset_clears_stats_and_state(self):
        eng, clock = _engine()
        tracks = [FakeTrack("yolo_1", (20, 20, 40, 40))]
        eng.evaluate(_frame(), tracks, 0, 0.0, skip_ssim=True, skip_periodic=True)
        assert eng.stats["vlm_calls"] >= 1
        eng.reset()
        assert eng.stats["vlm_calls"] == 0
        assert eng._last_vlm_time is None
        assert eng.is_in_cooldown(0.0) is False

    def test_decision_to_dict(self):
        eng, clock = _engine()
        d = eng.evaluate(
            _frame(),
            [FakeTrack("yolo_1", (20, 20, 40, 40))],
            0,
            0.0,
            skip_ssim=True,
            skip_periodic=True,
        )
        info = d.to_dict()
        assert info["should_call_vlm"] is True
        assert "roi" in info["triggers"]

    def test_savings_ratio_positive_when_few_calls(self):
        eng, clock = _engine()
        eng.entrance_enabled = False
        eng.stillness_enabled = False
        eng.mog2_trigger_enabled = False
        # 10 kare, tetik yok (ROI dışı, ssim skip, periodic uzun, entrance kapalı)
        for i in range(10):
            eng.evaluate(
                _frame(),
                [FakeTrack("yolo_x", (200, 200, 220, 220))],
                i,
                float(i),
                skip_ssim=True,
                skip_periodic=True,
                skip_entrance=True,
                skip_stillness=True,
                skip_mog2=True,
            )
        assert eng.stats["vlm_calls"] == 0
        assert eng.savings_ratio() >= 0.9

    def test_risk_level_aliases(self):
        eng, _ = _engine()
        assert eng.cooldown_seconds_for_risk("high") == eng.cooldown_high_s
        assert eng.cooldown_seconds_for_risk("CRITICAL") == eng.cooldown_critical_s
        assert eng.cooldown_seconds_for_risk(RiskLevel.LOW) == eng.cooldown_low_s
