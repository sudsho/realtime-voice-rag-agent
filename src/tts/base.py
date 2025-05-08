"""TTS provider interface."""

from __future__ import annotations

from typing import AsyncIterator, Protocol


class TtsProvider(Protocol):
    sample_rate: int

    def synth_stream(self, text: str) -> AsyncIterator[bytes]:
        """yield raw PCM16 audio chunks. caller owns framing for the wire."""
        ...
