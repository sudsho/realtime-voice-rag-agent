"""audio buffer + resample + chunk helpers.

browser sends opus or pcm16 at 48kHz. whisper wants 16kHz mono float32.
this module bridges that, plus exposes a chunk iterator that emits
fixed-size frames for VAD.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np


TARGET_SR = 16000
FRAME_MS = 30  # 30 ms frames are common for VAD


@dataclass
class AudioFrame:
    pcm: np.ndarray  # float32, mono
    sample_rate: int
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def pcm16_to_float32(buf: bytes) -> np.ndarray:
    """little-endian int16 bytes -> float32 in [-1, 1]."""
    arr = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
    arr /= 32768.0
    return arr


def float32_to_pcm16(arr: np.ndarray) -> bytes:
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16).tobytes()


def resample(arr: np.ndarray, src_sr: int, dst_sr: int = TARGET_SR) -> np.ndarray:
    """linear-interp resample. for streaming we don't need scipy/librosa.

    quality is fine for STT at 16k. for TTS playback use the native rate from
    the TTS provider so we never resample on the way out.
    """
    if src_sr == dst_sr:
        return arr
    if arr.size == 0:
        return arr
    ratio = dst_sr / src_sr
    n_out = int(round(arr.size * ratio))
    if n_out <= 1:
        return arr.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, num=arr.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, arr).astype(np.float32)


def to_mono(arr: np.ndarray, channels: int) -> np.ndarray:
    if channels == 1:
        return arr
    return arr.reshape(-1, channels).mean(axis=1).astype(np.float32)


class FrameChunker:
    """splits a streaming float32 buffer into fixed-ms frames."""

    def __init__(self, sample_rate: int = TARGET_SR, frame_ms: int = FRAME_MS):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_size = int(sample_rate * frame_ms / 1000)
        self._buf = np.zeros(0, dtype=np.float32)
        self._cursor_ms = 0

    def feed(self, samples: np.ndarray) -> list[AudioFrame]:
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        self._buf = np.concatenate([self._buf, samples])
        out: list[AudioFrame] = []
        while self._buf.size >= self.frame_size:
            frame = self._buf[: self.frame_size]
            self._buf = self._buf[self.frame_size :]
            out.append(AudioFrame(
                pcm=frame,
                sample_rate=self.sample_rate,
                start_ms=self._cursor_ms,
                end_ms=self._cursor_ms + self.frame_ms,
            ))
            self._cursor_ms += self.frame_ms
        return out

    def flush(self) -> AudioFrame | None:
        """flush any tail samples as a partial frame, padded with zeros."""
        if self._buf.size == 0:
            return None
        pad = np.zeros(self.frame_size - self._buf.size, dtype=np.float32)
        frame = np.concatenate([self._buf, pad])
        f = AudioFrame(
            pcm=frame,
            sample_rate=self.sample_rate,
            start_ms=self._cursor_ms,
            end_ms=self._cursor_ms + self.frame_ms,
        )
        self._buf = np.zeros(0, dtype=np.float32)
        self._cursor_ms += self.frame_ms
        return f
