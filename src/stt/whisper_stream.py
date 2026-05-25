"""wrapper around faster-whisper.

one transcribe call per user turn on the buffered utterance. does not do
its own windowing or partial-transcript emission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import numpy as np


@dataclass
class Transcript:
    text: str
    is_final: bool
    start_ms: int
    end_ms: int
    avg_logprob: float = 0.0


class WhisperStream:
    def __init__(
        self,
        model_name: str = "small.en",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
        language: str = "en",
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # type: ignore
        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(
        self,
        pcm_f32: np.ndarray,
        sample_rate: int = 16000,
        is_final: bool = True,
    ) -> Iterator[Transcript]:
        self._ensure_model()
        segments, info = self._model.transcribe(
            pcm_f32,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,  # VAD already done upstream
            condition_on_previous_text=False,
            without_timestamps=False,
        )
        for s in segments:
            yield Transcript(
                text=s.text.strip(),
                is_final=is_final,
                start_ms=int(s.start * 1000),
                end_ms=int(s.end * 1000),
                avg_logprob=getattr(s, "avg_logprob", 0.0) or 0.0,
            )
