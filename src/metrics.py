"""per-stage latency timers.

we annotate each turn with timestamps for: VAD-final, STT-final, retrieval,
LLM-first-token (TTFB), TTS-first-byte, audio-first-byte. then dump them as
structured logs for the latency notebook to plot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TurnMetrics:
    turn_id: str
    t_start: float = field(default_factory=time.perf_counter)
    marks: Dict[str, float] = field(default_factory=dict)

    def mark(self, label: str) -> None:
        self.marks[label] = time.perf_counter() - self.t_start

    def get_ms(self, label: str) -> Optional[float]:
        v = self.marks.get(label)
        return None if v is None else round(v * 1000.0, 2)

    def as_dict(self) -> Dict[str, float]:
        return {f"{k}_ms": round(v * 1000.0, 2) for k, v in self.marks.items()}

    def stage(self, a: str, b: str) -> Optional[float]:
        if a not in self.marks or b not in self.marks:
            return None
        return round((self.marks[b] - self.marks[a]) * 1000.0, 2)
