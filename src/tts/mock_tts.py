"""offline mock TTS.

drop-in replacement for the OpenAI / XTTS providers that needs no key, no
model, and no network. it turns text into deterministic synthetic PCM16 audio
bytes (a low-amplitude tone whose length scales with the text) and yields them
in ~80 ms frames, matching the streaming contract the pipeline expects.

the point is to exercise the LLM -> TTS -> WebSocket byte path, not to produce
intelligible speech.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator


class MockTts:
    sample_rate = 24000

    def __init__(self, voice: str = "mock", ms_per_char: float = 45.0, freq_hz: float = 180.0):
        self.voice = voice
        self.ms_per_char = ms_per_char
        self.freq_hz = freq_hz

    def _num_samples(self, text: str) -> int:
        ms = max(120.0, self.ms_per_char * max(1, len(text.strip())))
        return int(self.sample_rate * ms / 1000.0)

    async def synth_stream(self, text: str) -> AsyncIterator[bytes]:
        n = self._num_samples(text)
        frame = int(self.sample_rate * 0.08)  # ~80 ms frames
        two_pi_f = 2.0 * math.pi * self.freq_hz / self.sample_rate
        i = 0
        while i < n:
            buf = bytearray()
            end = min(i + frame, n)
            for s in range(i, end):
                # low-amplitude sine so the bytes are non-trivial but quiet
                val = int(6000.0 * math.sin(two_pi_f * s))
                buf += int(val).to_bytes(2, "little", signed=True)
            i = end
            yield bytes(buf)
