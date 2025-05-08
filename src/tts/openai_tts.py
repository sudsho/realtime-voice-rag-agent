"""openai streaming TTS.

we ask for raw 24kHz pcm and forward chunks straight to the websocket. the
browser decodes pcm16 into an AudioBuffer and queues it.
"""

from __future__ import annotations

import os
from typing import AsyncIterator


class OpenAiTts:
    sample_rate = 24000

    def __init__(self, voice: str = "alloy", model: str = "tts-1", chunk_bytes: int = 4096):
        self.voice = voice
        self.model = model
        self.chunk_bytes = chunk_bytes
        self._client = None

    def _ensure(self) -> None:
        if self._client is not None:
            return
        from openai import AsyncOpenAI  # type: ignore
        self._client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    async def synth_stream(self, text: str) -> AsyncIterator[bytes]:
        self._ensure()
        # openai sdk exposes a streaming response helper via with_streaming_response
        async with self._client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="pcm",
        ) as resp:
            async for chunk in resp.iter_bytes(self.chunk_bytes):
                if chunk:
                    yield chunk
