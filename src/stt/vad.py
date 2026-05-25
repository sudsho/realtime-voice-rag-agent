"""voice activity detection.

silero VAD via torch.hub, with a cheap RMS-energy fallback if the model
cannot be loaded. produces a stream of speech/silence transitions.

not currently wired into the serving path (see src/ws/server.py); kept here
as a utility that scripts and tests import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass
class VadEvent:
    kind: str  # "speech_start" | "speech_end" | "speech"
    start_ms: int
    end_ms: int


class SileroVad:
    """thin wrapper around the silero VAD model.

    we keep model loading lazy so importing this module is cheap.
    """

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._model = None
        self._utils = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
                onnx=False,
            )
            self._model = model
            self._utils = utils
        except Exception as e:  # noqa: BLE001
            # fall back to an energy-based heuristic if model not loadable
            self._model = _EnergyVad(self.threshold)

    def is_speech(self, frame: np.ndarray) -> bool:
        self._ensure_model()
        if isinstance(self._model, _EnergyVad):
            return self._model.is_speech(frame)
        import torch  # type: ignore
        with torch.no_grad():
            t = torch.from_numpy(frame).float()
            prob = float(self._model(t, self.sample_rate).item())
        return prob >= self.threshold


class _EnergyVad:
    """fallback. cheap RMS-energy check. not great but keeps things running."""

    def __init__(self, threshold: float = 0.5):
        # convert the prob threshold into an rms threshold roughly
        self.rms_thresh = max(0.005, 0.02 * threshold)

    def is_speech(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2) + 1e-12))
        return rms > self.rms_thresh


class UtteranceSegmenter:
    """consume per-frame VAD decisions, emit speech segments."""

    def __init__(
        self,
        min_silence_ms: int = 250,
        min_speech_ms: int = 200,
        pad_ms: int = 80,
    ):
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.pad_ms = pad_ms
        self._in_speech = False
        self._silence_run_ms = 0
        self._speech_run_ms = 0
        self._segment_start_ms: int | None = None

    def push(self, is_speech: bool, start_ms: int, end_ms: int) -> List[VadEvent]:
        out: List[VadEvent] = []
        dur = end_ms - start_ms
        if is_speech:
            self._silence_run_ms = 0
            self._speech_run_ms += dur
            if not self._in_speech and self._speech_run_ms >= self.min_speech_ms:
                self._in_speech = True
                self._segment_start_ms = max(0, start_ms - self.pad_ms - self._speech_run_ms)
                out.append(VadEvent("speech_start", self._segment_start_ms, start_ms))
        else:
            self._silence_run_ms += dur
            if self._in_speech and self._silence_run_ms >= self.min_silence_ms:
                self._in_speech = False
                seg_end = end_ms + self.pad_ms
                out.append(VadEvent("speech_end", self._segment_start_ms or 0, seg_end))
                self._speech_run_ms = 0
                self._segment_start_ms = None
        return out
