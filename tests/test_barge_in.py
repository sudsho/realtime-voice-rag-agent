"""tests for the barge-in detector."""

from __future__ import annotations

from src.agent.barge_in import BargeInDetector


def test_does_not_fire_below_trigger():
    d = BargeInDetector(trigger_ms=200)
    # 3 frames of 30ms each = 90ms, still below trigger
    for _ in range(3):
        assert d.push(True, frame_ms=30) is False


def test_fires_when_run_crosses_trigger():
    d = BargeInDetector(trigger_ms=200)
    fired = False
    for _ in range(8):  # 8 * 30 = 240ms
        if d.push(True, frame_ms=30):
            fired = True
            break
    assert fired is True


def test_silence_decays_run():
    d = BargeInDetector(trigger_ms=200)
    # build up some speech but not enough
    for _ in range(4):
        d.push(True, frame_ms=30)
    # silence frames should decrease counter
    for _ in range(5):
        assert d.push(False, frame_ms=30) is False
    assert d.state.speech_run_ms == 0


def test_reset_clears_state():
    d = BargeInDetector(trigger_ms=200)
    for _ in range(5):
        d.push(True, frame_ms=30)
    d.reset()
    assert d.state.speech_run_ms == 0
    assert d.state.last_seen == 0.0
