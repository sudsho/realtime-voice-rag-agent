"""offline mock STT.

drop-in replacement for :class:`src.stt.whisper_stream.WhisperStream` that
returns a canned transcript instead of loading faster-whisper. it lets the
full STT -> RAG -> LLM -> TTS pipeline run with no model download, no GPU, and
no ctranslate2/faster-whisper install.

the canned transcript stands in for what a real ASR pass would produce from the
synthetic audio bytes the caller buffered. the audio itself is inspected only
to report its duration; the text is fixed so the rest of the pipeline has a
deterministic turn to work with.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from src.stt.whisper_stream import Transcript

_DEFAULT_TRANSCRIPT = "How long do I have to get a refund?"


class MockStt:
    def __init__(self, transcript: str | None = None, sample_rate: int = 16000):
        self.transcript = (transcript or _DEFAULT_TRANSCRIPT).strip()
        self.sample_rate = sample_rate

    def transcribe(
        self,
        pcm_f32: np.ndarray,
        sample_rate: int = 16000,
        is_final: bool = True,
    ) -> Iterator[Transcript]:
        n = int(getattr(pcm_f32, "size", 0) or 0)
        dur_ms = int(round(1000.0 * n / (sample_rate or 16000)))
        yield Transcript(
            text=self.transcript,
            is_final=is_final,
            start_ms=0,
            end_ms=dur_ms,
            avg_logprob=-0.1,
        )
