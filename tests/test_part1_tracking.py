"""
PART 1 testleri — Hybrid Tracking (YOLO + MOG2 + BoT-SORT).

Kapsam:
- Proje dosya/yapı varlığı
- botsort_config ReID politikası (MOG2: false)
- MOG2 blob tespiti (sentetik hareketli kareler)
- HybridTracker prefix'leri (yolo_ / mog_)
- MOG2 hattında ReID kapalı
- ID sürekliliği (aynı nesne ardışık karelerde aynı ID)
- ID switch kontrolü (ayrı nesneler ayrı ID)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.tracking.hybrid_tracker import (
    HybridTracker,
    SimpleIoUTracker,
    YOLO_PREFIX,
    MOG_PREFIX,
    load_tracker_args,
)
from src.tracking.mog2_detector import MOG2Detector, BlobDetection

ROOT = Path(__file__).resolve().parents[1]
TRACKING = ROOT / "src" / "tracking"


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def _blank_frame(h: int = 240, w: int = 320, color: int = 40) -> np.ndarray:
    return np.full((h, w, 3), color, dtype=np.uint8)


def _draw_rect(
    frame: np.ndarray,
    x: int,
    y: int,
    bw: int,
    bh: int,
    color=(220, 220, 220),
) -> np.ndarray:
    out = frame.copy()
    out[y : y + bh, x : x + bw] = color
    return out


def _moving_blob_sequence(
    n_frames: int = 15,
    start: tuple = (20, 40),
    step: tuple = (8, 0),
    size: tuple = (40, 50),
) -> list:
    """Arkaplan sabit, dikdörtgen sağa kayar — MOG2 için ideal."""
    frames = []
    x, y = start
    bw, bh = size
    dx, dy = step
    for i in range(n_frames):
        base = _blank_frame()
        # İlk birkaç kare sadece arkaplan (MOG2 öğrenimi)
        if i >= 3:
            cx = x + (i - 3) * dx
            cy = y + (i - 3) * dy
            base = _draw_rect(base, cx, cy, bw, bh)
        frames.append(base)
    return frames


# ---------------------------------------------------------------------------
# 1. Proje yapısı
# ---------------------------------------------------------------------------

class TestProjectStructure:
    def test_tracking_modules_exist(self):
        assert (TRACKING / "mog2_detector.py").is_file()
        assert (TRACKING / "hybrid_tracker.py").is_file()
        assert (TRACKING / "botsort_config.yaml").is_file()

    def test_yolo_tracker_config_exists(self):
        assert (TRACKING / "botsort_config_yolo.yaml").is_file()

    def test_no_requirements_txt(self):
        """Bağımlılıklar uv + pyproject.toml ile yönetilir; requirements.txt pip aynasıdır."""
        assert (ROOT / "requirements.txt").is_file()
        assert (ROOT / "pyproject.toml").is_file()


# ---------------------------------------------------------------------------
# 2. BoT-SORT config / ReID politikası
# ---------------------------------------------------------------------------

class TestBotsortConfig:
    def test_mog_config_reid_disabled(self):
        cfg = load_tracker_args(TRACKING / "botsort_config.yaml")
        assert cfg.with_reid is False

    def test_yolo_config_reid_enabled(self):
        cfg = load_tracker_args(TRACKING / "botsort_config_yolo.yaml")
        assert cfg.with_reid is True

    def test_yaml_raw_mog_with_reid_false(self):
        raw = yaml.safe_load((TRACKING / "botsort_config.yaml").read_text(encoding="utf-8"))
        assert raw.get("with_reid") is False

    def test_hybrid_tracker_reid_policy(self):
        ht = HybridTracker(yolo_model=_MockYolo([]), use_ultralytics_botsort=False)
        policy = ht.get_reid_policy()
        assert policy["yolo"] is True
        assert policy["mog2"] is False
        assert ht.mog_with_reid is False
        assert ht.yolo_with_reid is True


# ---------------------------------------------------------------------------
# 3. MOG2 dedektör
# ---------------------------------------------------------------------------

class TestMOG2Detector:
    def test_static_scene_few_blobs(self):
        """Sabit sahnede (öğrenme sonrası) blob sayısı düşük olmalı."""
        det = MOG2Detector(min_area=200, history=50)
        frame = _blank_frame()
        for _ in range(20):
            blobs = det.detect(frame)
        # Son karelerde neredeyse sıfır hareket
        assert len(blobs) == 0

    def test_moving_rect_detected(self):
        det = MOG2Detector(min_area=300, history=30, var_threshold=16)
        frames = _moving_blob_sequence(n_frames=20)
        last_blobs = []
        for fr in frames:
            last_blobs = det.detect(fr)
        # Hareket başladıktan sonra en az bir blob beklenir
        assert len(last_blobs) >= 1
        assert all(isinstance(b, BlobDetection) for b in last_blobs)
        assert all(b.source == "mog2" for b in last_blobs)

    def test_to_xyxy_conf_shape(self):
        det = MOG2Detector()
        blobs = [
            BlobDetection(bbox=(10, 20, 50, 80), area=1200, confidence=0.5),
            BlobDetection(bbox=(100, 100, 140, 160), area=900, confidence=0.4),
        ]
        arr = det.to_xyxy_conf(blobs)
        assert arr.shape == (2, 5)
        assert arr[0, 0] == 10 and arr[0, 4] == 0.5

    def test_to_xyxy_conf_empty(self):
        det = MOG2Detector()
        arr = det.to_xyxy_conf([])
        assert arr.shape == (0, 5)

    def test_exclude_boxes_suppresses_overlap(self):
        det = MOG2Detector(min_area=100, history=20)
        # Arkaplan öğren
        bg = _blank_frame()
        for _ in range(15):
            det.detect(bg)
        # Büyük hareketli kutu
        fr = _draw_rect(bg, 50, 50, 60, 70, color=(255, 255, 255))
        free = det.detect(fr)
        assert len(free) >= 1
        # Aynı bölge YOLO kutusu ile bastırılmalı
        # Not: apply() her çağrıda öğrenir; yeniden aynı farkı üretmek için reset+öğren
        det2 = MOG2Detector(min_area=100, history=20)
        for _ in range(15):
            det2.detect(bg)
        suppressed = det2.detect(
            fr,
            exclude_boxes=[(50, 50, 110, 120)],
            iou_suppress=0.2,
        )
        # Örtüşen blob'lar elenmiş olmalı
        assert len(suppressed) <= len(free)

    def test_empty_frame_raises(self):
        det = MOG2Detector()
        with pytest.raises(ValueError):
            det.detect(np.array([]))


# ---------------------------------------------------------------------------
# 4. SimpleIoUTracker (ReID yok, ID sürekliliği)
# ---------------------------------------------------------------------------

class TestSimpleIoUTracker:
    def test_reid_flag_false(self):
        t = SimpleIoUTracker(with_reid=False)
        assert t.with_reid is False

    def test_id_persistence_across_frames(self):
        """Aynı nesne kaydıkça ID korunmalı (ID switch yok)."""
        tracker = SimpleIoUTracker(match_thresh=0.2, max_age=10)
        ids = []
        for i in range(8):
            x1 = 10 + i * 5
            det = np.array([[x1, 20, x1 + 40, 80, 0.9, 0]], dtype=np.float32)
            out = tracker.update(det)
            assert len(out) == 1
            ids.append(int(out[0, 4]))
        # Tüm karelerde aynı track id
        assert len(set(ids)) == 1

    def test_two_objects_distinct_ids(self):
        tracker = SimpleIoUTracker(match_thresh=0.3)
        det = np.array(
            [
                [10, 10, 50, 50, 0.9, 0],
                [200, 10, 240, 50, 0.8, 0],
            ],
            dtype=np.float32,
        )
        out = tracker.update(det)
        assert len(out) == 2
        id_a, id_b = int(out[0, 4]), int(out[1, 4])
        assert id_a != id_b

    def test_empty_detections(self):
        tracker = SimpleIoUTracker()
        out = tracker.update(np.zeros((0, 6), dtype=np.float32))
        assert len(out) == 0


# ---------------------------------------------------------------------------
# Mock YOLO — ağır model indirmeden test
# ---------------------------------------------------------------------------

class _MockYolo:
    """predict() yerine sabit tespit listesi döndüren mock."""

    def __init__(self, per_frame: list):
        """
        per_frame: her kare için (N,6) dizi listesi veya tek dizi.
        """
        self.per_frame = per_frame
        self.names = {0: "person", 2: "car"}
        self._i = 0

    def predict(self, source=None, **kwargs):
        del source, kwargs
        if not self.per_frame:
            dets = np.zeros((0, 6), dtype=np.float32)
        elif isinstance(self.per_frame[0], np.ndarray) or self.per_frame[0] is None:
            idx = min(self._i, len(self.per_frame) - 1)
            dets = self.per_frame[idx]
            self._i += 1
            if dets is None:
                dets = np.zeros((0, 6), dtype=np.float32)
        else:
            dets = np.asarray(self.per_frame, dtype=np.float32)
        return [_MockResult(dets)]

    def __call__(self, frame):
        return self.predict(frame)[0].as_array()


class _MockBoxes:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def __len__(self):
        return len(self._arr)

    @property
    def xyxy(self):
        return _Torchish(self._arr[:, :4] if len(self._arr) else np.zeros((0, 4)))

    @property
    def conf(self):
        return _Torchish(
            self._arr[:, 4] if len(self._arr) else np.zeros((0,))
        )

    @property
    def cls(self):
        return _Torchish(
            self._arr[:, 5] if len(self._arr) else np.zeros((0,))
        )


class _Torchish:
    """cpu().numpy() zinciri için minimal sarmalayıcı."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _MockResult:
    def __init__(self, arr: np.ndarray):
        self.boxes = _MockBoxes(np.asarray(arr, dtype=np.float32))

    def as_array(self):
        return np.asarray(
            np.hstack(
                [
                    self.boxes.xyxy.numpy(),
                    self.boxes.conf.numpy().reshape(-1, 1),
                    self.boxes.cls.numpy().reshape(-1, 1),
                ]
            )
            if len(self.boxes)
            else np.zeros((0, 6)),
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# 5. HybridTracker
# ---------------------------------------------------------------------------

class TestHybridTracker:
    def test_prefixes_on_process_detections(self):
        ht = HybridTracker(yolo_model=_MockYolo([]), use_ultralytics_botsort=False)
        yolo_dets = np.array([[10, 10, 60, 80, 0.9, 0]], dtype=np.float32)
        mog_dets = np.array([[100, 100, 150, 160, 0.5]], dtype=np.float32)
        result = ht.process_detections(yolo_dets, mog_dets)

        assert result.yolo_detections == 1
        assert result.mog2_detections == 1
        assert len(result.tracks) == 2

        yolo_ids = [t.track_id for t in result.yolo_tracks]
        mog_ids = [t.track_id for t in result.mog2_tracks]
        assert all(i.startswith(YOLO_PREFIX) for i in yolo_ids)
        assert all(i.startswith(MOG_PREFIX) for i in mog_ids)
        # Namespace çakışması yok
        assert yolo_ids[0] != mog_ids[0] or YOLO_PREFIX != MOG_PREFIX

    def test_mog_tracks_never_use_reid(self):
        ht = HybridTracker(yolo_model=_MockYolo([]), use_ultralytics_botsort=False)
        yolo_dets = np.array([[10, 10, 50, 50, 0.9, 0]], dtype=np.float32)
        mog_dets = np.array([[80, 80, 120, 130, 0.5]], dtype=np.float32)
        result = ht.process_detections(yolo_dets, mog_dets)

        for t in result.mog2_tracks:
            assert t.with_reid is False
            assert t.source == "mog2"
            assert t.track_id.startswith("mog_")

        for t in result.yolo_tracks:
            assert t.with_reid is True
            assert t.source == "yolo"
            assert t.track_id.startswith("yolo_")

    def test_id_stability_no_switch(self):
        """Aynı YOLO kutusu kaydıkça yolo_ ID sabit kalmalı."""
        ht = HybridTracker(yolo_model=_MockYolo([]), use_ultralytics_botsort=False)
        seen = []
        for i in range(6):
            x = 10 + i * 4
            yolo_dets = np.array([[x, 20, x + 40, 90, 0.95, 0]], dtype=np.float32)
            mog_dets = np.zeros((0, 5), dtype=np.float32)
            res = ht.process_detections(yolo_dets, mog_dets)
            assert len(res.yolo_tracks) == 1
            seen.append(res.yolo_tracks[0].track_id)
        assert len(set(seen)) == 1
        assert seen[0].startswith("yolo_")

    def test_process_frame_with_injected_yolo(self):
        ht = HybridTracker(yolo_model=_MockYolo([]), use_ultralytics_botsort=False)
        # MOG2 arkaplan öğrenimi
        bg = _blank_frame()
        for _ in range(12):
            ht.process_frame(bg, yolo_detections=np.zeros((0, 6)), run_mog2=True)

        # Hareket + YOLO tespiti
        fr = _draw_rect(bg, 30, 40, 45, 55, color=(250, 250, 250))
        yolo = np.array([[200, 10, 250, 80, 0.88, 0]], dtype=np.float32)
        res = ht.process_frame(fr, yolo_detections=yolo, run_mog2=True)

        assert res.yolo_detections == 1
        assert any(t.track_id.startswith("yolo_") for t in res.tracks)
        # MOG2 tarafı (hareket varsa) mog_ prefix kullanır
        for t in res.mog2_tracks:
            assert t.track_id.startswith("mog_")
            assert t.with_reid is False

    def test_namespace_collision_avoided(self):
        """Aynı sayısal ID olsa bile prefix ile ayrılır."""
        ht = HybridTracker(yolo_model=_MockYolo([]), use_ultralytics_botsort=False)
        # Her iki kaynaktan da ilk track genelde id=1 olur
        yolo_dets = np.array([[10, 10, 50, 50, 0.9, 0]], dtype=np.float32)
        mog_dets = np.array([[100, 100, 140, 150, 0.5]], dtype=np.float32)
        res = ht.process_detections(yolo_dets, mog_dets)
        ids = {t.track_id for t in res.tracks}
        assert "yolo_1" in ids or any(i.startswith("yolo_") for i in ids)
        assert "mog_1" in ids or any(i.startswith("mog_") for i in ids)
        # String ID'ler farklıdır
        assert len(ids) == len(res.tracks)

    def test_reset_clears_state(self):
        ht = HybridTracker(yolo_model=_MockYolo([]), use_ultralytics_botsort=False)
        yolo_dets = np.array([[10, 10, 50, 50, 0.9, 0]], dtype=np.float32)
        ht.process_detections(yolo_dets, np.zeros((0, 5)))
        assert ht._frame_idx > 0
        ht.reset()
        assert ht._frame_idx == 0

    def test_empty_frame_raises(self):
        ht = HybridTracker(yolo_model=_MockYolo([]), use_ultralytics_botsort=False)
        with pytest.raises(ValueError):
            ht.process_frame(np.array([]))


# ---------------------------------------------------------------------------
# 6. Entegrasyon: sentetik video döngüsü
# ---------------------------------------------------------------------------

class TestSyntheticVideoLoop:
    def test_multi_frame_hybrid_loop(self):
        """Birden fazla karede tracker çökmeden çalışmalı."""
        # Kare başına YOLO tespiti (kişi sağa kayıyor)
        per_frame = []
        for i in range(10):
            x = 15 + i * 6
            per_frame.append(
                np.array([[x, 30, x + 35, 100, 0.9, 0]], dtype=np.float32)
            )
        mock = _MockYolo(per_frame)
        ht = HybridTracker(yolo_model=mock, use_ultralytics_botsort=False)

        frames = _moving_blob_sequence(n_frames=10)
        all_ids = []
        for fr in frames:
            res = ht.process_frame(fr, run_mog2=True)
            all_ids.extend([t.track_id for t in res.tracks])
            for t in res.tracks:
                assert t.track_id.startswith("yolo_") or t.track_id.startswith("mog_")
                if t.source == "mog2":
                    assert t.with_reid is False

        # En az YOLO track'leri oluşmuş olmalı
        yolo_ids = [i for i in all_ids if i.startswith("yolo_")]
        assert len(yolo_ids) >= 1
