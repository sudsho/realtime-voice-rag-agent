"""pick the right TTS provider from config."""

from __future__ import annotations

from src.config import load_config


def make_tts(cfg: dict | None = None):
    cfg = cfg or load_config()
    tts_cfg = cfg.get("tts", {})
    provider = (tts_cfg.get("provider") or "openai").lower()
    voice = tts_cfg.get("voice", "alloy")
    if provider == "openai":
        from src.tts.openai_tts import OpenAiTts
        return OpenAiTts(voice=voice)
    if provider == "xtts":
        from src.tts.xtts import XttsProvider
        return XttsProvider(voice=voice)
    if provider in ("mock", "offline", "local"):
        from src.tts.mock_tts import MockTts
        return MockTts(voice=voice)
    raise ValueError(f"unknown tts provider: {provider}")
