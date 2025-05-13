"""per-connection session state.

each websocket connection owns a Session. it holds the audio buffer, VAD
state, current transcript, history, and latency metrics for the in-flight
turn. cancellation is handled by an asyncio.Event so we can yank a turn when
an "interrupt" message arrives mid-response.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.audio import FrameChunker, TARGET_SR
from src.metrics import TurnMetrics


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    client_sr: int = 48000
    chunker: FrameChunker = field(default_factory=lambda: FrameChunker(TARGET_SR, 30))
    history: List[dict] = field(default_factory=list)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    speech_buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    in_flight: Optional[TurnMetrics] = None

    def reset_speech(self) -> None:
        self.speech_buffer = np.zeros(0, dtype=np.float32)

    def append_speech(self, frame: np.ndarray) -> None:
        self.speech_buffer = np.concatenate([self.speech_buffer, frame])

    def add_user_turn(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})
        # keep history bounded for latency
        if len(self.history) > 12:
            self.history = self.history[-12:]

    def add_assistant_turn(self, text: str) -> None:
        if not text.strip():
            return
        self.history.append({"role": "assistant", "content": text})
        if len(self.history) > 12:
            self.history = self.history[-12:]

    def cancel(self) -> None:
        self.cancel_event.set()

    def reset_cancel(self) -> None:
        self.cancel_event = asyncio.Event()
