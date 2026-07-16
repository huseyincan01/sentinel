"""
PART 6 — Tek VLM mimarisi:

1) High-res YOLO/MOG2 (tracker full frame)
2) Tek VLM @ 336×336, periyot ~2 sn
3) Tetikte kırpma + %20 dolgu → yine 336 (high-res VLM yok)
"""

from __future__ import annotations

import inspect
import time

import numpy as np

from src.decision.triage_engine import TriggerKind, TriageEngine
from src.pipeline.main_pipeline import (
    SentinelPipeline,
    build_demo_pipeline,
    downscale_for_gate,
    downscale_for_vlm,
    get_motion_crop_box,
    crop_motion_patch,
)
from src.tracking.hybrid_tracker import FrameTracks, TrackedObject
from src.vlm.internvl_agent import (
    InternVLAgent,
    make_mock_generator,
)
from src.vlm.tools import ToolRegistry


def _track(
    tid="yolo_1",
    bbox=(20, 20, 50, 50),
    source="yolo",
    class_id=0,
    class_name="person",
):
    return TrackedObject(
        track_id=tid,
        bbox=bbox,
        confidence=0.9,
        class_id=class_id,
        class_name=class_name,
        source=source,
        frame_idx=0,
        with_reid=source == "yolo",
    )


class DummyTracker:
    def __init__(self, tracks=None):
        self.default = tracks or []
        self.seen_shapes = []

    def reset(self):
        pass

    def process_frame(self, frame, yolo_detections=None, run_mog2=True):
        self.seen_shapes.append(frame.shape[:2])
        return FrameTracks(
            frame_idx=0,
            tracks=list(self.default),
            yolo_detections=0,
            mog2_detections=0,
        )


class TestTrackerHighRes:
    def test_tracker_gets_full_frame_not_336(self, tmp_path):
        tracker = DummyTracker([])
        tools = ToolRegistry(report_dir=tmp_path)
        agent = InternVLAgent(
            generator_fn=make_mock_generator(),
            tools=tools,
            auto_execute_tools=False,
        )
        pipe = SentinelPipeline(
            tracker=tracker,
            triage=TriageEngine(
                entrance_enabled=False,
                stillness_enabled=False,
                mog2_trigger_enabled=False,
                motion_enabled=False,
                color_fire_enabled=False,
                periodic_interval_s=999,
            ),
            agent=agent,
            tools=tools,
            report_dir=tmp_path,
        )
        big = np.zeros((480, 640, 3), dtype=np.uint8)
        pipe.process_frame(big, frame_idx=0, timestamp_s=0.0)
        assert tracker.seen_shapes[-1] == (480, 640)


class TestSingleVlm336:
    def test_downscale_336(self):
        fr = np.zeros((720, 1280, 3), dtype=np.uint8)
        low = downscale_for_vlm(fr, 336)
        assert low.shape == (336, 336, 3)
        assert downscale_for_gate(fr, 336).shape == (336, 336, 3)

    def test_vlm_always_336_never_high_res(self, tmp_path):
        seen = {}

        def detail_gen(prompt="", image=None, **kw):
            if image is not None and hasattr(image, "shape"):
                seen["shape"] = image.shape
            return make_mock_generator()(prompt=prompt, image=image)

        tracker = DummyTracker([_track()])
        tools = ToolRegistry(report_dir=tmp_path)
        agent = InternVLAgent(
            generator_fn=detail_gen,
            tools=tools,
            auto_execute_tools=False,
        )
        pipe = SentinelPipeline(
            tracker=tracker,
            triage=TriageEngine(
                entrance_enabled=True,
                stillness_enabled=False,
                mog2_trigger_enabled=False,
                motion_enabled=False,
                color_fire_enabled=False,
                periodic_interval_s=999,
            ),
            agent=agent,
            tools=tools,
            report_dir=tmp_path,
            vlm_size=336,
        )
        big = np.zeros((200, 300, 3), dtype=np.uint8)
        fr = pipe.process_frame(big, frame_idx=0, timestamp_s=0.0)
        assert fr.vlm_called is True
        assert fr.vlm_input_shape == (336, 336, 3)
        assert fr.detail_high_res_shape == (336, 336, 3)
        assert seen.get("shape") == (336, 336, 3)

    def test_vlm_period_2s(self, tmp_path):
        tracker = DummyTracker([])
        tools = ToolRegistry(report_dir=tmp_path)
        n = {"c": 0}

        def gen(**kw):
            n["c"] += 1
            return make_mock_generator()(**kw)

        agent = InternVLAgent(
            generator_fn=gen,
            tools=tools,
            auto_execute_tools=False,
        )
        pipe = SentinelPipeline(
            tracker=tracker,
            triage=TriageEngine(
                entrance_enabled=False,
                stillness_enabled=False,
                mog2_trigger_enabled=False,
                motion_enabled=False,
                color_fire_enabled=False,
                periodic_interval_s=999,
            ),
            agent=agent,
            tools=tools,
            report_dir=tmp_path,
            vlm_period_s=2.0,
            save_reports=False,
        )
        # 0, 0.5, 1.0 → yalnızca 0; 2.0 → ikinci
        for i, t in enumerate([0.0, 0.5, 1.0, 2.0]):
            pipe.process_frame(
                np.zeros((80, 80, 3), dtype=np.uint8), frame_idx=i, timestamp_s=t
            )
        assert n["c"] == 2

    def test_async_worker_waits_for_model(self, tmp_path):
        """Worker model bitmeden yenisini basmaz; period ≥ 2s veya inference."""

        class EmptyTracker:
            def reset(self):
                pass

            def process_frame(self, frame, yolo_detections=None, run_mog2=True):
                return FrameTracks(
                    frame_idx=0, tracks=[], yolo_detections=0, mog2_detections=0
                )

        tools = ToolRegistry(report_dir=tmp_path)
        times = []

        def slow_gen(**kw):
            times.append(time.monotonic())
            time.sleep(0.15)
            return make_mock_generator()(**kw)

        agent = InternVLAgent(
            generator_fn=slow_gen,
            tools=tools,
            auto_execute_tools=False,
        )
        pipe = SentinelPipeline(
            tracker=EmptyTracker(),
            triage=TriageEngine(
                entrance_enabled=False,
                stillness_enabled=False,
                mog2_trigger_enabled=False,
                motion_enabled=False,
                color_fire_enabled=False,
                periodic_interval_s=999,
            ),
            agent=agent,
            tools=tools,
            report_dir=tmp_path,
            vlm_period_s=0.4,  # test için kısa period
            save_reports=False,
        )
        pipe.process_frame(
            np.zeros((64, 64, 3), dtype=np.uint8),
            frame_idx=0,
            fps=30.0,
            timestamp_s=0.0,
            run_async=True,
        )
        pipe.start_vlm_worker()
        time.sleep(1.2)
        pipe.stop_vlm_worker()
        assert pipe._vlm_call_count >= 2
        # Ardışık çağrılar en az ~0.15s (inference) aralıklı
        if len(times) >= 2:
            gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
            assert min(gaps) >= 0.12


class TestBypass:
    def test_stillness_still_runs_vlm(self, tmp_path):
        tracker = DummyTracker([_track(bbox=(10, 10, 30, 30))])
        tools = ToolRegistry(report_dir=tmp_path)
        agent = InternVLAgent(
            generator_fn=make_mock_generator(),
            tools=tools,
            auto_execute_tools=False,
        )
        pipe = SentinelPipeline(
            tracker=tracker,
            triage=TriageEngine(
                entrance_enabled=False,
                stillness_enabled=True,
                stillness_duration_s=1.0,
                stillness_max_move_px=5.0,
                mog2_trigger_enabled=False,
                motion_enabled=False,
                color_fire_enabled=False,
                periodic_interval_s=999,
            ),
            agent=agent,
            tools=tools,
            report_dir=tmp_path,
        )
        frame = np.full((100, 100, 3), 40, dtype=np.uint8)
        last = None
        for i, t in enumerate([0.0, 0.4, 0.8, 1.2]):
            last = pipe.process_frame(frame, frame_idx=i, timestamp_s=t)
        assert last is not None
        assert pipe._vlm_call_count >= 1





class TestPrepareImage:
    def test_no_or_true_in_body(self):
        src = inspect.getsource(InternVLAgent._prepare_image)
        lines = []
        in_doc = False
        for line in src.splitlines():
            if '"""' in line:
                cnt = line.count('"""')
                if cnt >= 2:
                    continue
                in_doc = not in_doc
                continue
            if in_doc:
                continue
            lines.append(line)
        assert "or True" not in "\n".join(lines)


class TestDemo:
    def test_build_demo(self, tmp_path):
        pipe = build_demo_pipeline(
            report_dir=tmp_path, mock_vlm=True, force_gate_every_candidate=True
        )
        assert pipe.vlm_size == 336
        assert abs(pipe.vlm_period_s - 2.0) < 1e-6
        r = pipe.process_frames(
            [np.full((60, 80, 3), 30, dtype=np.uint8) for _ in range(3)],
            fps=1.0,
            timestamps=[0.0, 1.0, 2.0],
        )
        assert r.success is True


class TestColorFireAndTemporalCrop:
    def test_color_fire_trigger(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[10:30, 10:30] = [0, 0, 255]
        triage = TriageEngine(color_fire_enabled=True, color_fire_threshold_px=150)
        ev = triage.check_color_fire(frame, frame_idx=0, timestamp=0.0)
        assert ev is not None
        assert ev.kind == TriggerKind.COLOR_FIRE
        assert ev.details["fire_pixels"] >= 400
        bundle = triage.collect_candidates(frame, frame_idx=0, timestamp=0.0)
        assert any(t.kind == TriggerKind.COLOR_FIRE for t in bundle.triggers)

    def test_get_motion_crop_box(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        box = get_motion_crop_box(frame, mask, min_area=200, padding_pct=0.2)
        assert box is not None
        x, y, w, h = box
        assert x < 40 and y < 40
        assert x + w > 60 and y + h > 60
        cropped = crop_motion_patch(frame, box)
        assert cropped.shape == (h, w, 3)
