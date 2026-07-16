"""
PART 4 testleri — Ana Pipeline entegrasyonu.

Mock tracker + mock VLM ile uçtan uca video akışı (model indirmeden).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytest

from src.decision.triage_engine import TriageEngine
from src.pipeline.main_pipeline import (
    SentinelPipeline,
    build_demo_pipeline,
)
from src.tracking.hybrid_tracker import FrameTracks, HybridTracker, TrackedObject
from src.vlm.internvl_agent import InternVLAgent, make_mock_generator
from src.vlm.tools import ToolRegistry


# ---------------------------------------------------------------------------
# Dummy bileşenler
# ---------------------------------------------------------------------------

class DummyTracker:
    """Her karede sabit veya programlanabilir track üretir."""

    def __init__(
        self,
        tracks_per_frame: Optional[List[List[TrackedObject]]] = None,
        default_tracks: Optional[List[TrackedObject]] = None,
    ):
        self.tracks_per_frame = tracks_per_frame
        self.default_tracks = default_tracks or []
        self._i = 0
        self.reset_count = 0

    def reset(self) -> None:
        self._i = 0
        self.reset_count += 1

    def process_frame(self, frame, yolo_detections=None, run_mog2=True) -> FrameTracks:
        del yolo_detections, run_mog2
        if self.tracks_per_frame is not None and self._i < len(self.tracks_per_frame):
            tracks = self.tracks_per_frame[self._i]
        else:
            tracks = list(self.default_tracks)
        idx = self._i
        self._i += 1
        return FrameTracks(frame_idx=idx, tracks=list(tracks), yolo_detections=0, mog2_detections=0)


def _track(
    tid: str = "yolo_1",
    bbox: Tuple[int, int, int, int] = (20, 20, 50, 50),
    source: str = "yolo",
) -> TrackedObject:
    return TrackedObject(
        track_id=tid,
        bbox=bbox,
        confidence=0.9,
        class_id=0,
        class_name="person",
        source=source,
        frame_idx=0,
        with_reid=(source == "yolo"),
    )


def _frames(n: int = 10, h: int = 80, w: int = 100, seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        # Hafif değişim — SSIM çoğu zaman tetiklemez
        base = np.full((h, w, 3), 40 + (i % 3), dtype=np.uint8)
        if i == 5:
            # Ani sahne değişimi (SSIM tetik adayı)
            base = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        out.append(base)
    return out


def _write_test_video(path: Path, n: int = 8, fps: float = 10.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = 64, 96
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert writer.isOpened(), "VideoWriter açılamadı"
    for i in range(n):
        frame = np.full((h, w, 3), 30 + i * 5, dtype=np.uint8)
        cv2.rectangle(frame, (10 + i, 10), (30 + i, 40), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    return path


# ROI sol üst
ROI = [(0, 0), (60, 0), (60, 60), (0, 60)]


def _pipeline(tmp_path: Path, tracker=None, periodic_s: float = 100.0, **kw) -> SentinelPipeline:
    tools = ToolRegistry(report_dir=tmp_path)
    agent = InternVLAgent(
        generator_fn=make_mock_generator(),
        tools=tools,
        auto_execute_tools=True,
    )
    triage = TriageEngine(
        roi_polygon=ROI,
        periodic_interval_s=periodic_s,
        coalesce_window_ms=500.0,
        cooldown_low_s=2.0,
        ssim_threshold=0.85,
        ssim_severe_threshold=0.4,
    )
    force = kw.pop("force_gate_every_candidate", True)
    period_s = kw.pop("vlm_period_s", 2.0)
    return SentinelPipeline(
        tracker=tracker or DummyTracker(default_tracks=[]),
        triage=triage,
        agent=agent,
        tools=tools,
        report_dir=tmp_path,
        save_reports=True,
        store_frame_results=True,
        force_gate_every_candidate=force,
        vlm_size=336,
        vlm_period_s=period_s,
        **kw,
    )


# ---------------------------------------------------------------------------
# Temel akış
# ---------------------------------------------------------------------------

class TestPipelineBasic:
    def test_process_frames_completes(self, tmp_path):
        pipe = _pipeline(tmp_path)
        frames = _frames(12)
        result = pipe.process_frames(frames, fps=10.0, source="synthetic")
        assert result.success is True
        assert result.error is None
        assert result.frames_processed == 12

    def test_roi_triggers_vlm(self, tmp_path):
        # Track ROI içinde → VLM
        tracker = DummyTracker(default_tracks=[_track("yolo_1", (10, 10, 40, 40))])
        pipe = _pipeline(tmp_path, tracker=tracker, periodic_s=999)
        frames = _frames(5)
        # Zaman damgaları: cooldown'u aşmak için aralıklı
        timestamps = [0.0, 3.0, 6.0, 9.0, 12.0]
        result = pipe.process_frames(frames, fps=1.0, timestamps=timestamps)
        assert result.success
        assert result.vlm_calls >= 1
        assert result.last_analysis is not None
        assert result.last_summary

    def test_no_track_no_roi_few_vlm(self, tmp_path):
        """Tek VLM 2 sn periyot → 0.8 sn'lik 8 karede az çağrı."""
        tracker = DummyTracker(default_tracks=[_track("yolo_9", (200, 200, 220, 220))])
        pipe = _pipeline(
            tmp_path,
            tracker=tracker,
            periodic_s=999,
            force_gate_every_candidate=False,
            vlm_period_s=2.0,
        )
        frames = [np.full((80, 100, 3), 50, dtype=np.uint8) for _ in range(8)]
        # fps=10 → 8 kare = 0.8 sn video zamanı → ~1 VLM (period 2s)
        result = pipe.process_frames(frames, fps=10.0)
        assert result.success
        assert result.vlm_calls < result.frames_processed
        assert result.vlm_calls <= 2

    def test_reports_written_on_vlm(self, tmp_path):
        tracker = DummyTracker(default_tracks=[_track("yolo_1", (15, 15, 35, 35))])
        pipe = _pipeline(tmp_path, tracker=tracker)
        pipe._force_next_incident = True
        frames = _frames(3)
        timestamps = [0.0, 5.0, 10.0]
        result = pipe.process_frames(frames, fps=1.0, timestamps=timestamps)
        if result.vlm_calls > 0:
            assert len(result.reports) > 0
            for p in result.reports:
                path = Path(p)
                assert path.is_file()
                data = json.loads(path.read_text(encoding="utf-8"))
                assert "summary" in data
                assert "risk" in data
                assert "pipeline_meta" in data

    def test_tools_invoked_when_vlm_returns_tools(self, tmp_path):
        high = {
            "summary": "Forklift kazası ve yaralanma riski.",
            "events": [
                {"time": "00:01", "event": "Forklift devrildi", "severity": "Yüksek"}
            ],
            "risk": "Yüksek",
            "risk_score": 0.9,
            "actions": ["Ambulans çağır"],
            "tools_called": ["call_ambulance", "lock_area", "trigger_alarm"],
            "frame_analyzed": 0,
        }
        tools = ToolRegistry(report_dir=tmp_path)
        agent = InternVLAgent(
            generator_fn=make_mock_generator(high),
            tools=tools,
            auto_execute_tools=True,
        )
        tracker = DummyTracker(default_tracks=[_track("yolo_1", (10, 10, 40, 40))])
        pipe = SentinelPipeline(
            tracker=tracker,
            triage=TriageEngine(roi_polygon=ROI, periodic_interval_s=999),
            agent=agent,
            tools=tools,
            report_dir=tmp_path,
        )
        pipe._force_next_incident = True
        result = pipe.process_frames(_frames(1), fps=1.0, timestamps=[0.0])
        assert result.vlm_calls >= 1
        names = [c.tool for c in tools.call_history]
        assert "call_ambulance" in names
        assert "lock_area" in names

    def test_on_frame_callback(self, tmp_path):
        seen = []
        pipe = _pipeline(
            tmp_path,
            tracker=DummyTracker(default_tracks=[_track()]),
        )

        def cb(fr, frame):
            seen.append(fr.frame_idx)
            assert frame is not None

        pipe.process_frames(_frames(4), fps=2.0, timestamps=[0, 3, 6, 9], on_frame=cb)
        assert seen == [0, 1, 2, 3]

    def test_empty_frame_raises(self, tmp_path):
        pipe = _pipeline(tmp_path)
        with pytest.raises(ValueError):
            pipe.process_frame(np.array([]))

    def test_reset_between_videos(self, tmp_path):
        tracker = DummyTracker(default_tracks=[_track()])
        pipe = _pipeline(tmp_path, tracker=tracker)
        pipe.process_frames(_frames(2), fps=1.0, timestamps=[0.0, 5.0])
        assert tracker.reset_count >= 1
        pipe.process_frames(_frames(2), fps=1.0, timestamps=[0.0, 5.0])
        assert tracker.reset_count >= 2


# ---------------------------------------------------------------------------
# OpenCV video dosyası
# ---------------------------------------------------------------------------

class TestProcessVideoFile:
    def test_process_video_mp4(self, tmp_path):
        video_path = _write_test_video(tmp_path / "clip.mp4", n=6, fps=10.0)
        tracker = DummyTracker(default_tracks=[])
        pipe = _pipeline(tmp_path, tracker=tracker, periodic_s=999)
        result = pipe.process_video(str(video_path), max_frames=6)
        assert result.success is True
        assert result.frames_processed == 6
        assert result.error is None

    def test_process_video_missing_file(self, tmp_path):
        pipe = _pipeline(tmp_path)
        result = pipe.process_video(str(tmp_path / "yok.mp4"))
        assert result.success is False
        assert result.error is not None

    def test_iter_video(self, tmp_path):
        video_path = _write_test_video(tmp_path / "iter.mp4", n=4, fps=8.0)
        pipe = _pipeline(
            tmp_path,
            tracker=DummyTracker(default_tracks=[]),
            periodic_s=999,
        )
        pairs = list(pipe.iter_video(str(video_path), max_frames=4))
        assert len(pairs) == 4
        assert pairs[0][0].frame_idx == 0


# ---------------------------------------------------------------------------
# İstatistik / entegrasyon
# ---------------------------------------------------------------------------

class TestPipelineStats:
    def test_stats_contain_savings(self, tmp_path):
        frames = [np.full((60, 80, 3), 50, dtype=np.uint8) for _ in range(20)]
        pipe = _pipeline(
            tmp_path,
            tracker=DummyTracker(default_tracks=[_track("yolo_x", (200, 200, 210, 210))]),
            periodic_s=999,
        )
        result = pipe.process_frames(frames, fps=10.0)
        assert "vlm_call_ratio" in result.stats
        assert "vlm_savings_approx" in result.stats
        assert result.stats["vlm_savings_approx"] >= 0.0

    def test_to_dict_serializable(self, tmp_path):
        pipe = _pipeline(
            tmp_path,
            tracker=DummyTracker(default_tracks=[_track()]),
        )
        result = pipe.process_frames(_frames(2), fps=1.0, timestamps=[0.0, 5.0])
        d = result.to_dict()
        # json serileştirilebilir olmalı
        raw = json.dumps(d, ensure_ascii=False, default=str)
        assert "frames_processed" in raw

    def test_get_last_summary_and_report(self, tmp_path):
        pipe = _pipeline(
            tmp_path,
            tracker=DummyTracker(default_tracks=[_track()]),
        )
        pipe.process_frames(_frames(2), fps=1.0, timestamps=[0.0, 5.0])
        if pipe.last_pipeline_result and pipe.last_pipeline_result.vlm_calls > 0:
            assert isinstance(pipe.get_last_summary(), str)
            rep = pipe.get_last_report_json()
            assert rep is not None
            assert "summary" in rep

    def test_build_demo_pipeline(self, tmp_path):
        pipe = build_demo_pipeline(report_dir=tmp_path, roi_polygon=ROI, mock_vlm=True)
        assert pipe.tracker is not None
        assert pipe.agent is not None
        result = pipe.process_frames(_frames(5), fps=5.0)
        assert result.success is True

    def test_hybrid_tracker_injected_detections(self, tmp_path):
        """Gerçek HybridTracker + enjekte YOLO tespitleri (model yok)."""
        ht = HybridTracker(
            yolo_model=build_demo_pipeline(report_dir=tmp_path).tracker._yolo,
            use_ultralytics_botsort=False,
        )
        tools = ToolRegistry(report_dir=tmp_path)
        agent = InternVLAgent(
            generator_fn=make_mock_generator(),
            tools=tools,
            auto_execute_tools=False,
        )
        pipe = SentinelPipeline(
            tracker=ht,
            triage=TriageEngine(roi_polygon=ROI, periodic_interval_s=999),
            agent=agent,
            tools=tools,
            report_dir=tmp_path,
        )
        frames = _frames(4)
        # ROI içinde kayan kutu
        dets = []
        for i in range(4):
            x = 10 + i * 2
            dets.append(np.array([[x, 10, x + 30, 40, 0.9, 0]], dtype=np.float32))
        result = pipe.process_frames(
            frames,
            fps=1.0,
            timestamps=[0.0, 3.0, 6.0, 9.0],
            yolo_detections_per_frame=dets,
        )
        assert result.success
        assert result.frames_processed == 4
        # En az bir karede track oluşmuş olmalı
        tracked = [fr for fr in result.frame_results if fr.tracks]
        assert len(tracked) >= 1

    def test_vlm_failure_does_not_crash_pipeline(self, tmp_path):
        """VLM bozuk JSON döndürse bile pipeline ayakta kalır."""
        tools = ToolRegistry(report_dir=tmp_path)
        agent = InternVLAgent(
            generator_fn=lambda **kw: "%%%degil-json%%%",
            tools=tools,
            auto_execute_tools=True,
        )
        pipe = SentinelPipeline(
            tracker=DummyTracker(default_tracks=[_track()]),
            triage=TriageEngine(roi_polygon=ROI, periodic_interval_s=999),
            agent=agent,
            tools=tools,
            report_dir=tmp_path,
        )
        pipe._force_next_incident = True
        result = pipe.process_frames(_frames(2), fps=1.0, timestamps=[0.0, 5.0])
        assert result.success is True
        if result.vlm_calls:
            assert result.last_analysis is not None
