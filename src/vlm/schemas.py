"""
VLM çıktı şeması — şartname uyumlu Pydantic modelleri.

lm-format-enforcer ve runtime doğrulama bu şemayı kullanır.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


RiskLiteral = Literal["Düşük", "Orta", "Yüksek", "Kritik"]


class EventItem(BaseModel):
    """Tek bir olay kaydı."""

    time: str = Field(..., description="MM:SS veya HH:MM:SS zaman damgası")
    event: str = Field(..., description="Olay açıklaması (Türkçe)")
    severity: RiskLiteral = Field(..., description="Olay ciddiyet seviyesi")

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, v: Any) -> str:
        if v is None:
            return "Düşük"
        mapping = {
            "düşük": "Düşük",
            "dusuk": "Düşük",
            "low": "Düşük",
            "orta": "Orta",
            "medium": "Orta",
            "yüksek": "Yüksek",
            "yuksek": "Yüksek",
            "high": "Yüksek",
            "kritik": "Kritik",
            "critical": "Kritik",
        }
        key = str(v).strip().lower()
        return mapping.get(key, str(v).strip() if str(v).strip() in {
            "Düşük", "Orta", "Yüksek", "Kritik"
        } else "Düşük")

    @field_validator("time")
    @classmethod
    def validate_time(cls, v: str) -> str:
        v = str(v).strip()
        parts = v.split(":")
        if len(parts) not in (2, 3):
            raise ValueError("time formatı MM:SS veya HH:MM:SS olmalı")
        for p in parts:
            if not p.isdigit():
                raise ValueError("time sayısal olmalı")
        return v


class AnalysisResult(BaseModel):
    """
    Zorunlu VLM JSON çıktısı (şartname).

    Örnek:
    {
      "summary": "...",
      "events": [...],
      "risk": "Yüksek",
      "risk_score": 0.87,
      "actions": [...],
      "tools_called": ["call_ambulance", "lock_area"],
      "timestamp": "2026-07-15T00:10:00",
      "frame_analyzed": 450
    }
    """

    summary: str = Field(..., description="Türkçe özet")
    events: List[EventItem] = Field(default_factory=list)
    risk: RiskLiteral = Field(..., description="Genel risk seviyesi")
    risk_score: float = Field(..., ge=0.0, le=1.0)
    actions: List[str] = Field(default_factory=list)
    tools_called: List[str] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    frame_analyzed: int = Field(0, ge=0)

    @field_validator("risk", mode="before")
    @classmethod
    def normalize_risk(cls, v: Any) -> str:
        if v is None:
            return "Düşük"
        mapping = {
            "düşük": "Düşük",
            "dusuk": "Düşük",
            "low": "Düşük",
            "orta": "Orta",
            "medium": "Orta",
            "yüksek": "Yüksek",
            "yuksek": "Yüksek",
            "high": "Yüksek",
            "kritik": "Kritik",
            "critical": "Kritik",
        }
        key = str(v).strip().lower()
        return mapping.get(key, str(v).strip() if str(v).strip() in {
            "Düşük", "Orta", "Yüksek", "Kritik"
        } else mapping.get(key, "Düşük"))

    @field_validator("risk_score", mode="before")
    @classmethod
    def clamp_score(cls, v: Any) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = 0.0
        return max(0.0, min(1.0, f))

    @field_validator("actions", "tools_called", mode="before")
    @classmethod
    def coerce_to_list(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        if isinstance(v, str):
            return [v]
        return []

    def to_memory_line(self) -> str:
        """Bellek prompt satırı."""
        t = self.events[0].time if self.events else "--:--"
        ev = self.events[0].event if self.events else self.summary[:80]
        return f"[{t}] {ev} (risk={self.risk})"

    def model_dump_report(self) -> Dict[str, Any]:
        return self.model_dump()


def analysis_json_schema() -> Dict[str, Any]:
    """Detail VLM için JSON Schema (lm-format-enforcer)."""
    return AnalysisResult.model_json_schema()


UrgencyLiteral = Literal["low", "medium", "high", "critical"]


class GateDecision(BaseModel):
    """
    Gate (hakem) VLM çıktısı — high-res detail gerekli mi?

    Low-res VLM bu şemayı üretir; tools çağırmaz.
    """

    need_high_res: bool = Field(..., description="High-res detaylı analiz gerekli mi?")
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    reason: str = Field("", description="Kısa Türkçe gerekçe")
    urgency: UrgencyLiteral = Field("medium")

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_conf(cls, v: Any) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = 0.5
        return max(0.0, min(1.0, f))

    @field_validator("urgency", mode="before")
    @classmethod
    def norm_urgency(cls, v: Any) -> str:
        if v is None:
            return "medium"
        key = str(v).strip().lower()
        mapping = {
            "low": "low",
            "düşük": "low",
            "dusuk": "low",
            "medium": "medium",
            "orta": "medium",
            "high": "high",
            "yüksek": "high",
            "yuksek": "high",
            "critical": "critical",
            "kritik": "critical",
        }
        return mapping.get(key, "medium")


def gate_json_schema() -> Dict[str, Any]:
    return GateDecision.model_json_schema()


KNOWN_TOOLS = [
    "call_ambulance",
    "alert_security_team",
    "lock_area",
    "generate_incident_report",
    "notify_supervisor",
    "trigger_alarm",
]
