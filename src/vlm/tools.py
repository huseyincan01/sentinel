"""
Mock araçlar (Tools) — Sentinel ajanı için simüle müdahale fonksiyonları.

Gerçek harici sistem çağrısı YAPMAZ; konsola log basar, sonucu dict olarak
döndürür ve JSON raporuna yazılmak üzere kayıt tutar.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger("sentinel.tools")

# Varsayılan rapor dizini
DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "reports"

# Alarm seviyeleri
ALARM_LEVELS = frozenset({"low", "medium", "high", "critical"})


@dataclass
class ToolResult:
    """Tek bir araç çağrısının sonucu."""

    tool: str
    success: bool
    message: str
    args: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    simulated: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ToolRegistry:
    """
    Mock araç kaydı ve çalıştırıcısı.

    Tüm çağrılar `call_history` içinde tutulur (rapor / test için).
    """

    def __init__(self, report_dir: Optional[Path] = None) -> None:
        self.report_dir = Path(report_dir) if report_dir else DEFAULT_REPORT_DIR
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.call_history: List[ToolResult] = []
        self._handlers: Dict[str, Callable[..., ToolResult]] = {
            "call_ambulance": self.call_ambulance,
            "alert_security_team": self.alert_security_team,
            "lock_area": self.lock_area,
            "generate_incident_report": self.generate_incident_report,
            "notify_supervisor": self.notify_supervisor,
            "trigger_alarm": self.trigger_alarm,
        }

    def reset_history(self) -> None:
        self.call_history.clear()

    def available_tools(self) -> List[str]:
        return sorted(self._handlers.keys())

    def _record(self, result: ToolResult) -> ToolResult:
        self.call_history.append(result)
        level = logging.INFO if result.success else logging.WARNING
        logger.log(
            level,
            "[MOCK TOOL] %s | success=%s | %s | args=%s",
            result.tool,
            result.success,
            result.message,
            result.args,
        )
        # Konsola da net log
        print(
            f"[MOCK TOOL] {result.tool} | success={result.success} | {result.message}"
        )
        return result

    # ------------------------------------------------------------------
    # 6 mock araç
    # ------------------------------------------------------------------

    def call_ambulance(self, location: str) -> ToolResult:
        """Tıbbi acil çağrısı simülasyonu."""
        loc = str(location).strip() or "bilinmeyen konum"
        return self._record(
            ToolResult(
                tool="call_ambulance",
                success=True,
                message=f"Ambulans çağrısı simüle edildi: {loc}",
                args={"location": loc},
            )
        )

    def alert_security_team(self, message: str) -> ToolResult:
        """Güvenlik ekibine uyarı."""
        msg = str(message).strip() or "Genel güvenlik uyarısı"
        return self._record(
            ToolResult(
                tool="alert_security_team",
                success=True,
                message=f"Güvenlik ekibi bilgilendirildi: {msg}",
                args={"message": msg},
            )
        )

    def lock_area(self, zone_id: str) -> ToolResult:
        """Belirtilen bölgeyi kapat / erişimi engelle."""
        zone = str(zone_id).strip() or "unknown_zone"
        return self._record(
            ToolResult(
                tool="lock_area",
                success=True,
                message=f"Bölge kilitlendi (simülasyon): {zone}",
                args={"zone_id": zone},
            )
        )

    def generate_incident_report(self, data: dict) -> ToolResult:
        """JSON raporu dosyaya kaydet."""
        if not isinstance(data, dict):
            return self._record(
                ToolResult(
                    tool="generate_incident_report",
                    success=False,
                    message="data dict olmalı",
                    args={"data_type": type(data).__name__},
                )
            )
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = self.report_dir / f"incident_{ts}.json"
        payload = dict(data)
        payload.setdefault("report_generated_at", datetime.now(timezone.utc).isoformat())
        try:
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return self._record(
                ToolResult(
                    tool="generate_incident_report",
                    success=True,
                    message=f"Rapor kaydedildi: {path}",
                    args={"path": str(path), "keys": list(payload.keys())},
                )
            )
        except OSError as exc:
            return self._record(
                ToolResult(
                    tool="generate_incident_report",
                    success=False,
                    message=f"Rapor yazılamadı: {exc}",
                    args={"path": str(path)},
                )
            )

    def notify_supervisor(self, message: str) -> ToolResult:
        """Amir / yöneticiye bildirim."""
        msg = str(message).strip() or "Olay bildirimi"
        return self._record(
            ToolResult(
                tool="notify_supervisor",
                success=True,
                message=f"Amire bildirim gönderildi (simülasyon): {msg}",
                args={"message": msg},
            )
        )

    def trigger_alarm(self, level: str) -> ToolResult:
        """Sesli/görsel alarm (low/medium/high/critical)."""
        lvl = str(level).strip().lower()
        if lvl not in ALARM_LEVELS:
            return self._record(
                ToolResult(
                    tool="trigger_alarm",
                    success=False,
                    message=f"Geçersiz alarm seviyesi: {level}. "
                    f"İzin verilen: {sorted(ALARM_LEVELS)}",
                    args={"level": level},
                )
            )
        return self._record(
            ToolResult(
                tool="trigger_alarm",
                success=True,
                message=f"Alarm tetiklendi (simülasyon): seviye={lvl}",
                args={"level": lvl},
            )
        )

    # ------------------------------------------------------------------
    # Çalıştırma
    # ------------------------------------------------------------------

    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """İsimle araç çalıştır."""
        handler = self._handlers.get(tool_name)
        if handler is None:
            return self._record(
                ToolResult(
                    tool=tool_name,
                    success=False,
                    message=f"Bilinmeyen araç: {tool_name}",
                    args=kwargs,
                )
            )
        try:
            return handler(**kwargs)
        except TypeError as exc:
            return self._record(
                ToolResult(
                    tool=tool_name,
                    success=False,
                    message=f"Argüman hatası: {exc}",
                    args=kwargs,
                )
            )

    def execute_from_analysis(
        self,
        tools_called: Sequence[str],
        report_data: Optional[dict] = None,
        location: str = "fabrika_alani",
        zone_id: str = "zone_1",
        supervisor_message: Optional[str] = None,
        security_message: Optional[str] = None,
        alarm_level: str = "high",
    ) -> List[ToolResult]:
        """
        Analiz çıktısındaki tools_called listesine göre araçları sırayla çalıştır.

        Bilinen araç adları için makul varsayılan argümanlar kullanılır.
        """
        results: List[ToolResult] = []
        summary = ""
        if report_data:
            summary = str(report_data.get("summary", ""))
        for name in tools_called:
            key = str(name).strip()
            if key == "call_ambulance":
                results.append(self.call_ambulance(location))
            elif key == "alert_security_team":
                results.append(
                    self.alert_security_team(
                        security_message or summary or "Güvenlik uyarısı"
                    )
                )
            elif key == "lock_area":
                results.append(self.lock_area(zone_id))
            elif key == "generate_incident_report":
                results.append(
                    self.generate_incident_report(report_data or {"summary": summary})
                )
            elif key == "notify_supervisor":
                results.append(
                    self.notify_supervisor(
                        supervisor_message or summary or "Olay bildirimi"
                    )
                )
            elif key == "trigger_alarm":
                # risk alanından seviye türetmeye çalış
                lvl = alarm_level
                if report_data:
                    risk = str(report_data.get("risk", "")).lower()
                    if "kritik" in risk or "critical" in risk:
                        lvl = "critical"
                    elif "yüksek" in risk or "yuksek" in risk or "high" in risk:
                        lvl = "high"
                    elif "orta" in risk or "medium" in risk:
                        lvl = "medium"
                    elif "düşük" in risk or "dusuk" in risk or "low" in risk:
                        lvl = "low"
                results.append(self.trigger_alarm(lvl))
            else:
                results.append(self.execute(key))
        return results


# Modül düzeyinde kolay API (AGENTS.md imzaları)
_default_registry = ToolRegistry()


def call_ambulance(location: str) -> ToolResult:
    return _default_registry.call_ambulance(location)


def alert_security_team(message: str) -> ToolResult:
    return _default_registry.alert_security_team(message)


def lock_area(zone_id: str) -> ToolResult:
    return _default_registry.lock_area(zone_id)


def generate_incident_report(data: dict) -> ToolResult:
    return _default_registry.generate_incident_report(data)


def notify_supervisor(message: str) -> ToolResult:
    return _default_registry.notify_supervisor(message)


def trigger_alarm(level: str) -> ToolResult:
    return _default_registry.trigger_alarm(level)


def get_default_registry() -> ToolRegistry:
    return _default_registry
