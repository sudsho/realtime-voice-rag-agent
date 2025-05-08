"""coqui XTTS-v2 fallback.

we run XTTS in a worker thread because the model is heavy and not natively
async. quality is good but cold start is ~5s on CPU. only used when the
openai provider is unavailable.
"""

from __future__ import annotations

import asyncio
import io
from typing import AsyncIterator, Optional

import numpy as np


class XttsProvider:
    sample_rate = 24000

    def __init__(self, voice: str = "default", language: str = "en", model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        self.voice = voice
        self.language = language
        self.model_name = model_name
        self._tts = None
        self._lock = asyncio.Lock()
        self._speaker_wav: Optional[str] = None

    def _ensure(self) -> None:
        if self._tts is not None:
            return
        # imported lazily; coqui-tts is a heavy dep
        from TTS.api import TTS  # type: ignore
        self._tts = TTS(self.model_name)

    def _synth_blocking(self, text: str) -> np.ndarray:
        self._ensure()
        wav = self._tts.tts(
            text=text,
            language=self.language,
            speaker_wav=self._speaker_wav,
        )
        return np.asarray(wav, dtype=np.float32)

    async def synth_stream(self, text: str) -> AsyncIterator[bytes]:
        # XTTS is not natively streaming; we synthesize in a thread and chunk
        # the output to keep the perceived latency low.
        loop = asyncio.get_running_loop()
        async with self._lock:
            wav = await loop.run_in_executor(None, self._synth_blocking, text)
        # interleave-friendly chunks of ~80ms
        chunk_n = int(self.sample_rate * 0.08)
        for i in range(0, len(wav), chunk_n):
            block = wav[i : i + chunk_n]
            block = np.clip(block, -1.0, 1.0)
            pcm = (block * 32767.0).astype(np.int16).tobytes()
            yield pcm
            await asyncio.sleep(0)
