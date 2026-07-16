"""
İkili ajan belleği (Memory).

1) Kayan pencere (sliding window): rutin olaylar, son N=10
2) Sticky (kalıcı) liste: Yüksek/Kritik risk, maks M=5
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, List, Optional, Sequence, Union

from src.vlm.schemas import AnalysisResult


HIGH_RISKS = frozenset({"Yüksek", "Kritik", "high", "critical", "yuksek", "yüksek"})


@dataclass
class MemoryEvent:
    """Bellekte tutulan sade olay."""

    time: str
    event: str
    risk: str
    frame_analyzed: int = 0
    summary: str = ""

    @classmethod
    def from_analysis(cls, result: AnalysisResult) -> "MemoryEvent":
        if result.events:
            t = result.events[0].time
            ev = result.events[0].event
        else:
            t = "--:--"
            ev = result.summary[:120]
        return cls(
            time=t,
            event=ev,
            risk=result.risk,
            frame_analyzed=result.frame_analyzed,
            summary=result.summary,
        )

    def format_line(self) -> str:
        return f"[{self.time}] {self.event} (risk={self.risk})"

    def is_high_risk(self) -> bool:
        return str(self.risk).strip() in HIGH_RISKS or str(self.risk).strip().lower() in {
            "yüksek",
            "yuksek",
            "kritik",
            "high",
            "critical",
        }


class AgentMemory:
    """
    İkili bellek yöneticisi.

    - window: son N rutin olay (deque, maxlen=N)
    - sticky: Yüksek/Kritik olaylar (maks M, en eski düşer)
    """

    def __init__(self, window_size: int = 10, sticky_size: int = 5) -> None:
        if window_size < 1:
            raise ValueError("window_size >= 1 olmalı")
        if sticky_size < 1:
            raise ValueError("sticky_size >= 1 olmalı")
        self.window_size = window_size
        self.sticky_size = sticky_size
        self._window: Deque[MemoryEvent] = deque(maxlen=window_size)
        self._sticky: List[MemoryEvent] = []

    def reset(self) -> None:
        self._window.clear()
        self._sticky.clear()

    @property
    def recent_events(self) -> List[MemoryEvent]:
        return list(self._window)

    @property
    def sticky_events(self) -> List[MemoryEvent]:
        return list(self._sticky)

    def __len__(self) -> int:
        return len(self._window)

    def add(self, item: Union[AnalysisResult, MemoryEvent, dict]) -> MemoryEvent:
        """Yeni olayı belleğe ekle."""
        event = self._coerce(item)
        self._window.append(event)
        if event.is_high_risk():
            self._sticky.append(event)
            # Maks M — en eski kritik düşer (memory leak önleme)
            if len(self._sticky) > self.sticky_size:
                self._sticky = self._sticky[-self.sticky_size :]
        return event

    def add_many(self, items: Iterable[Union[AnalysisResult, MemoryEvent, dict]]) -> None:
        for it in items:
            self.add(it)

    @staticmethod
    def _coerce(item: Union[AnalysisResult, MemoryEvent, dict]) -> MemoryEvent:
        if isinstance(item, MemoryEvent):
            return item
        if isinstance(item, AnalysisResult):
            return MemoryEvent.from_analysis(item)
        if isinstance(item, dict):
            events = item.get("events") or []
            if events and isinstance(events[0], dict):
                t = str(events[0].get("time", "--:--"))
                ev = str(events[0].get("event", item.get("summary", "")))
            else:
                t = "--:--"
                ev = str(item.get("summary", item.get("event", "")))[:120]
            return MemoryEvent(
                time=t,
                event=ev,
                risk=str(item.get("risk", "Düşük")),
                frame_analyzed=int(item.get("frame_analyzed", 0)),
                summary=str(item.get("summary", "")),
            )
        raise TypeError(f"Desteklenmeyen bellek öğesi: {type(item)}")

    def build_context_prompt(self) -> str:
        """
        VLM prompt eki.

        Örnek:
        "Kritik Geçmiş: [00:15] Forklift devrildi. Son Olaylar: [00:18] Kişi yürüyor."
        """
        sticky_part = "Yok"
        if self._sticky:
            sticky_part = " ".join(e.format_line() for e in self._sticky)

        recent_part = "Yok"
        if self._window:
            recent_part = " ".join(e.format_line() for e in self._window)

        return (
            f"Kritik Geçmiş: {sticky_part}. "
            f"Son Olaylar: {recent_part}."
        )

    def as_dict(self) -> dict:
        return {
            "window_size": self.window_size,
            "sticky_size": self.sticky_size,
            "recent": [
                {
                    "time": e.time,
                    "event": e.event,
                    "risk": e.risk,
                    "frame_analyzed": e.frame_analyzed,
                }
                for e in self._window
            ],
            "sticky": [
                {
                    "time": e.time,
                    "event": e.event,
                    "risk": e.risk,
                    "frame_analyzed": e.frame_analyzed,
                }
                for e in self._sticky
            ],
        }
