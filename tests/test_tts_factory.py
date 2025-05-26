"""tests for the TTS factory dispatch (no real network calls)."""

from __future__ import annotations

import pytest

from src.tts import factory


def test_unknown_provider_raises():
    cfg = {"tts": {"provider": "made-up"}}
    with pytest.raises(ValueError):
        factory.make_tts(cfg)


def test_default_provider_is_openai(monkeypatch):
    cfg = {"tts": {}}
    captured = {}

    class FakeOpenAi:
        def __init__(self, voice):
            captured["voice"] = voice

    monkeypatch.setattr("src.tts.openai_tts.OpenAiTts", FakeOpenAi)
    inst = factory.make_tts(cfg)
    assert isinstance(inst, FakeOpenAi)
    assert captured["voice"] == "alloy"


def test_xtts_provider(monkeypatch):
    cfg = {"tts": {"provider": "xtts", "voice": "narrator"}}
    captured = {}

    class FakeXtts:
        def __init__(self, voice):
            captured["voice"] = voice

    monkeypatch.setattr("src.tts.xtts.XttsProvider", FakeXtts)
    inst = factory.make_tts(cfg)
    assert isinstance(inst, FakeXtts)
    assert captured["voice"] == "narrator"
