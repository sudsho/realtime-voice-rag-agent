"""pick the STT backend from config.

``stt.backend`` selects between the real faster-whisper wrapper (default) and
the offline mock. the mock backend needs no model download, no GPU, and no
faster-whisper install, so the demo pipeline runs anywhere.
"""

from __future__ import annotations


def make_stt(cfg: dict | None = None):
    cfg = cfg or {}
    stt_cfg = cfg.get("stt", {})
    backend = (stt_cfg.get("backend") or "faster_whisper").lower()
    if backend in ("mock", "offline"):
        from src.stt.mock_stt import MockStt

        return MockStt(transcript=stt_cfg.get("mock_transcript"))
    from src.stt.whisper_stream import WhisperStream

    return WhisperStream(
        model_name=stt_cfg.get("model", "small.en"),
        device=stt_cfg.get("device", "cpu"),
        compute_type=stt_cfg.get("compute_type", "int8"),
        beam_size=int(stt_cfg.get("beam_size", 1)),
        language=stt_cfg.get("language", "en"),
    )
