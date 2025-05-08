"""LLM streaming client. openai chat completions, streaming."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import AsyncIterator, List


@dataclass
class LlmDelta:
    text: str
    finished: bool = False


class StreamingLlm:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 256,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _ensure(self) -> None:
        if self._client is not None:
            return
        from openai import AsyncOpenAI  # type: ignore
        self._client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    async def stream(
        self,
        messages: List[dict],
    ) -> AsyncIterator[LlmDelta]:
        self._ensure()
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", None) or ""
                finish_reason = getattr(chunk.choices[0], "finish_reason", None)
            except (IndexError, AttributeError):
                continue
            if token:
                yield LlmDelta(text=token, finished=False)
            if finish_reason:
                yield LlmDelta(text="", finished=True)
                break
