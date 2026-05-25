"""barge-in detector.

utility class that, given a stream of VAD decisions, flags the moment a
sustained speech run crosses ``trigger_ms``. intended for cancelling an
in-flight turn when the user starts talking again.

not wired into the serving path in this repo: no VAD runs on the inbound
mic stream in src/ws/server.py, and the shipped frontend does not send an
interrupt message. the server does accept `{"type":"interrupt"}` if a
client chooses to send one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class BargeInState:
    speech_run_ms: int = 0
    last_seen: float = 0.0


class BargeInDetector:
    def __init__(self, trigger_ms: int = 200):
        self.trigger_ms = trigger_ms
        self.state = BargeInState()

    def push(self, is_speech: bool, frame_ms: int) -> bool:
        """returns True the moment the speech run crosses ``trigger_ms``."""
        now = time.monotonic()
        if is_speech:
            if self.state.last_seen and (now - self.state.last_seen) < 0.5:
                self.state.speech_run_ms += frame_ms
            else:
                self.state.speech_run_ms = frame_ms
            self.state.last_seen = now
            return self.state.speech_run_ms >= self.trigger_ms
        # silence - decay
        self.state.speech_run_ms = max(0, self.state.speech_run_ms - frame_ms)
        return False

    def reset(self) -> None:
        self.state = BargeInState()
