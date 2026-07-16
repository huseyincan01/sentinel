"""
PART 3 testleri — VLM Agent, Mock Tools, JSON şema, Memory.

Model indirmeden mock generator ile çalışır.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from src.vlm.internvl_agent import (
    InternVLAgent,
    build_analysis_prompt,
    extract_json_object,
    make_mock_generator,
)
from src.vlm.memory import AgentMemory, MemoryEvent
from src.vlm.schemas import AnalysisResult, EventItem, KNOWN_TOOLS, analysis_json_schema
from src.vlm.tools import (
    ToolRegistry,
    alert_security_team,
    call_ambulance,
    generate_incident_report,
    lock_area,
    notify_supervisor,
    trigger_alarm,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class TestMockTools:
    def test_all_six_tools_registered(self, tmp_path):
        reg = ToolRegistry(report_dir=tmp_path)
        names = set(reg.available_tools())
        expected = {
            "call_ambulance",
            "alert_security_team",
            "lock_area",
            "generate_incident_report",
            "notify_supervisor",
            "trigger_alarm",
        }
        assert expected == names

    def test_call_ambulance(self, tmp_path):
        reg = ToolRegistry(report_dir=tmp_path)
        r = reg.call_ambulance("hat_3")
        assert r.success and r.tool == "call_ambulance"
        assert r.args["location"] == "hat_3"
        assert r.simulated is True

    def test_alert_security_team(self, tmp_path):
        r = ToolRegistry(report_dir=tmp_path).alert_security_team("İzinsiz giriş")
        assert r.success and "İzinsiz" in r.message

    def test_lock_area(self, tmp_path):
        r = ToolRegistry(report_dir=tmp_path).lock_area("zone_A")
        assert r.success and r.args["zone_id"] == "zone_A"

    def test_generate_incident_report_writes_file(self, tmp_path):
        reg = ToolRegistry(report_dir=tmp_path)
        r = reg.generate_incident_report({"summary": "test olay", "risk": "Yüksek"})
        assert r.success
        files = list(tmp_path.glob("incident_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["summary"] == "test olay"

    def test_notify_supervisor(self, tmp_path):
        r = ToolRegistry(report_dir=tmp_path).notify_supervisor("Acil durum")
        assert r.success

    def test_trigger_alarm_valid_levels(self, tmp_path):
        reg = ToolRegistry(report_dir=tmp_path)
        for lvl in ("low", "medium", "high", "critical"):
            assert reg.trigger_alarm(lvl).success

    def test_trigger_alarm_invalid(self, tmp_path):
        r = ToolRegistry(report_dir=tmp_path).trigger_alarm("extreme")
        assert r.success is False

    def test_module_level_functions(self):
        # Global registry — en azından çağrılabilir
        assert call_ambulance("x").tool == "call_ambulance"
        assert alert_security_team("m").success
        assert lock_area("z1").success
        assert notify_supervisor("n").success
        assert trigger_alarm("high").success

    def test_execute_from_analysis(self, tmp_path):
        reg = ToolRegistry(report_dir=tmp_path)
        results = reg.execute_from_analysis(
            ["call_ambulance", "lock_area", "trigger_alarm"],
            report_data={"summary": "Kaza", "risk": "Kritik"},
            location="depo",
            zone_id="z9",
        )
        assert len(results) == 3
        assert all(r.success for r in results)
        assert results[2].args["level"] == "critical"

    def test_unknown_tool(self, tmp_path):
        r = ToolRegistry(report_dir=tmp_path).execute("teleport")
        assert r.success is False


# ---------------------------------------------------------------------------
# Schemas / JSON
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_valid_analysis_result(self):
        r = AnalysisResult(
            summary="Forklift kazası gözlendi.",
            events=[
                EventItem(time="00:15", event="Forklift devrildi", severity="Yüksek"),
            ],
            risk="Yüksek",
            risk_score=0.87,
            actions=["Sağlık ekibini çağır"],
            tools_called=["call_ambulance", "lock_area"],
            frame_analyzed=450,
        )
        assert r.risk == "Yüksek"
        assert r.risk_score == 0.87
        d = r.model_dump_report()
        assert "summary" in d and "tools_called" in d

    def test_risk_normalization(self):
        r = AnalysisResult(
            summary="x",
            risk="high",
            risk_score=1.5,  # clamp
            frame_analyzed=0,
        )
        assert r.risk == "Yüksek"
        assert r.risk_score == 1.0

    def test_invalid_time_rejected(self):
        with pytest.raises(ValidationError):
            EventItem(time="ab:cd", event="x", severity="Düşük")

    def test_json_schema_has_required_keys(self):
        schema = analysis_json_schema()
        assert "properties" in schema
        props = schema["properties"]
        for key in (
            "summary",
            "events",
            "risk",
            "risk_score",
            "actions",
            "tools_called",
            "timestamp",
            "frame_analyzed",
        ):
            assert key in props

    def test_extract_json_object_from_fence(self):
        text = 'Önce metin\n```json\n{"summary": "ok", "risk": "Düşük", "risk_score": 0.1}\n```\n'
        obj = extract_json_object(text)
        assert obj["summary"] == "ok"

    def test_known_tools_list(self):
        assert len(KNOWN_TOOLS) == 6


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class TestAgentMemory:
    def test_sliding_window_maxlen(self):
        mem = AgentMemory(window_size=10, sticky_size=5)
        for i in range(15):
            mem.add(
                {
                    "summary": f"olay {i}",
                    "risk": "Düşük",
                    "events": [{"time": f"00:{i:02d}", "event": f"e{i}", "severity": "Düşük"}],
                    "frame_analyzed": i,
                }
            )
        assert len(mem.recent_events) == 10
        # En eski 0..4 düşmüş olmalı; son 5..14
        assert mem.recent_events[0].event == "e5"
        assert mem.recent_events[-1].event == "e14"

    def test_sticky_only_high_critical(self):
        mem = AgentMemory(window_size=10, sticky_size=5)
        mem.add({"summary": "a", "risk": "Düşük", "events": [{"time": "00:01", "event": "rutin", "severity": "Düşük"}]})
        mem.add({"summary": "b", "risk": "Yüksek", "events": [{"time": "00:02", "event": "kaza", "severity": "Yüksek"}]})
        mem.add({"summary": "c", "risk": "Kritik", "events": [{"time": "00:03", "event": "yaralı", "severity": "Kritik"}]})
        assert len(mem.sticky_events) == 2
        assert mem.sticky_events[0].event == "kaza"
        assert mem.sticky_events[1].event == "yaralı"

    def test_sticky_maxlen(self):
        mem = AgentMemory(window_size=10, sticky_size=5)
        for i in range(7):
            mem.add(
                {
                    "summary": f"kritik {i}",
                    "risk": "Kritik",
                    "events": [
                        {"time": f"00:{i:02d}", "event": f"crit{i}", "severity": "Kritik"}
                    ],
                }
            )
        assert len(mem.sticky_events) == 5
        assert mem.sticky_events[0].event == "crit2"
        assert mem.sticky_events[-1].event == "crit6"

    def test_build_context_prompt_format(self):
        mem = AgentMemory()
        mem.add(
            {
                "summary": "Forklift",
                "risk": "Yüksek",
                "events": [
                    {"time": "00:15", "event": "Forklift devrildi", "severity": "Yüksek"}
                ],
            }
        )
        mem.add(
            {
                "summary": "yürüme",
                "risk": "Düşük",
                "events": [
                    {"time": "00:18", "event": "Kişi yürüyor", "severity": "Düşük"}
                ],
            }
        )
        ctx = mem.build_context_prompt()
        assert "Kritik Geçmiş:" in ctx
        assert "Son Olaylar:" in ctx
        assert "Forklift devrildi" in ctx
        assert "Kişi yürüyor" in ctx
        assert "00:15" in ctx

    def test_empty_memory_prompt(self):
        mem = AgentMemory()
        ctx = mem.build_context_prompt()
        assert "Yok" in ctx

    def test_from_analysis_result(self):
        ar = AnalysisResult(
            summary="test",
            events=[EventItem(time="01:00", event="X", severity="Orta")],
            risk="Orta",
            risk_score=0.4,
            frame_analyzed=10,
        )
        mem = AgentMemory()
        mem.add(ar)
        assert mem.recent_events[0].event == "X"


# ---------------------------------------------------------------------------
# InternVL Agent (mock)
# ---------------------------------------------------------------------------

class TestInternVLAgentMock:
    def test_analyze_returns_valid_schema(self, tmp_path):
        agent = InternVLAgent(
            generator_fn=make_mock_generator(),
            tools=ToolRegistry(report_dir=tmp_path),
            auto_execute_tools=False,
        )
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = agent.analyze(frame=frame, frame_idx=10, fps=30.0)
        assert isinstance(result, AnalysisResult)
        assert result.frame_analyzed == 10
        assert 0.0 <= result.risk_score <= 1.0

    def test_memory_added_to_prompt(self, tmp_path):
        agent = InternVLAgent(
            generator_fn=make_mock_generator(),
            tools=ToolRegistry(report_dir=tmp_path),
            auto_execute_tools=False,
        )
        # Yüksek riskli olay enjekte et
        high = {
            "summary": "Forklift kazası",
            "events": [
                {"time": "00:15", "event": "Forklift devrildi", "severity": "Yüksek"}
            ],
            "risk": "Yüksek",
            "risk_score": 0.9,
            "actions": [],
            "tools_called": [],
            "frame_analyzed": 1,
        }
        agent.memory.add(high)
        prompt = agent.build_prompt(frame_idx=100)
        assert "Kritik Geçmiş:" in prompt
        assert "Forklift devrildi" in prompt
        assert "Son Olaylar:" in prompt
        assert "Şimdi bu kareyi analiz et" in prompt

    def test_prompt_includes_system_and_memory(self):
        ctx = "Kritik Geçmiş: [00:15] X. Son Olaylar: [00:18] Y."
        p = build_analysis_prompt(ctx, frame_idx=30, fps=30.0)
        assert "Sentinel" in p
        assert "Kritik Geçmiş" in p
        assert "frame=30" in p

    def test_tools_auto_executed(self, tmp_path):
        fixed = {
            "summary": "Kaza",
            "events": [
                {"time": "00:10", "event": "Düşme", "severity": "Kritik"}
            ],
            "risk": "Kritik",
            "risk_score": 0.95,
            "actions": ["Ambulans"],
            "tools_called": ["call_ambulance", "trigger_alarm", "notify_supervisor"],
            "frame_analyzed": 5,
        }
        reg = ToolRegistry(report_dir=tmp_path)
        agent = InternVLAgent(
            generator_fn=make_mock_generator(fixed),
            tools=reg,
            auto_execute_tools=True,
        )
        result = agent.analyze(frame_idx=5)
        assert result.tools_called == fixed["tools_called"]
        assert len(agent.last_tool_results) == 3
        assert all(r.success for r in agent.last_tool_results)
        assert len(reg.call_history) == 3

    def test_parse_invalid_json_fallback(self, tmp_path):
        agent = InternVLAgent(
            generator_fn=lambda **kw: "bu json değil!!!",
            tools=ToolRegistry(report_dir=tmp_path),
            auto_execute_tools=False,
        )
        result = agent.analyze(frame_idx=1)
        assert isinstance(result, AnalysisResult)
        assert "manuel" in result.summary.lower() or result.risk in (
            "Orta",
            "Düşük",
            "Yüksek",
            "Kritik",
        )

    def test_high_risk_mock_from_prompt_keyword(self, tmp_path):
        """make_mock_generator prompt'ta anahtar görünce yüksek risk üretir."""
        captured = {}

        def gen(prompt: str = "", image=None, **kw):
            captured["prompt"] = prompt
            # Bilerek forklift kelimesi prompt'a girmese bile fixed kullanmıyoruz;
            # generator kendi kararını versin — ekstra ile zorla
            return make_mock_generator()(prompt="forklift kaza " + prompt, image=image)

        agent = InternVLAgent(
            generator_fn=gen,
            tools=ToolRegistry(report_dir=tmp_path),
            auto_execute_tools=False,
        )
        result = agent.analyze(frame_idx=450, trigger_info="ROI")
        assert result.risk in ("Yüksek", "Kritik")
        assert "call_ambulance" in result.tools_called or result.risk_score >= 0.5

    def test_load_with_mock_skips_download(self):
        agent = InternVLAgent(generator_fn=make_mock_generator())
        agent.load()
        assert agent.is_loaded

    def test_format_enforcer_parser_builds(self, tmp_path):
        agent = InternVLAgent(
            generator_fn=make_mock_generator(),
            tools=ToolRegistry(report_dir=tmp_path),
        )
        parser = agent.build_format_enforcer_parser()
        # lm-format-enforcer kurulu olmalı
        assert parser is not None

    def test_memory_grows_across_analyze_calls(self, tmp_path):
        agent = InternVLAgent(
            generator_fn=make_mock_generator(),
            tools=ToolRegistry(report_dir=tmp_path),
            auto_execute_tools=False,
            window_size=10,
        )
        for i in range(3):
            agent.analyze(frame_idx=i)
        assert len(agent.memory.recent_events) == 3
        ctx = agent.get_memory_prompt()
        assert "Son Olaylar:" in ctx

    def test_reset_memory(self, tmp_path):
        agent = InternVLAgent(
            generator_fn=make_mock_generator(),
            tools=ToolRegistry(report_dir=tmp_path),
            auto_execute_tools=False,
        )
        agent.analyze(frame_idx=0)
        agent.reset_memory()
        assert len(agent.memory.recent_events) == 0

    def test_analyze_with_tracks_in_prompt(self, tmp_path):
        seen = {}

        def gen(prompt: str = "", image=None, **kw):
            seen["prompt"] = prompt
            return make_mock_generator()(prompt=prompt, image=image)

        agent = InternVLAgent(
            generator_fn=gen,
            tools=ToolRegistry(report_dir=tmp_path),
            auto_execute_tools=False,
        )
        tracks = [
            {"track_id": "yolo_1", "source": "yolo"},
            {"track_id": "mog_2", "source": "mog2"},
        ]
        agent.analyze(frame_idx=3, tracks=tracks)
        assert "yolo_1" in seen["prompt"]
        assert "mog_2" in seen["prompt"]
