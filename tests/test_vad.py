import numpy as np

from src.stt.vad import UtteranceSegmenter, _EnergyVad


def test_energy_vad_detects_loud_signal():
    vad = _EnergyVad(threshold=0.5)
    loud = np.full(480, 0.5, dtype=np.float32)
    quiet = np.full(480, 0.001, dtype=np.float32)
    assert vad.is_speech(loud) is True
    assert vad.is_speech(quiet) is False


def test_segmenter_emits_speech_start_and_end():
    seg = UtteranceSegmenter(min_silence_ms=60, min_speech_ms=60, pad_ms=0)
    events = []
    # 6 frames of speech then 6 frames of silence (extra margin so the
    # threshold check fires reliably even if frame_ms aliasing nudges)
    t = 0
    for _ in range(6):
        events += seg.push(True, t, t + 30)
        t += 30
    for _ in range(6):
        events += seg.push(False, t, t + 30)
        t += 30
    kinds = [e.kind for e in events]
    assert "speech_start" in kinds
    assert "speech_end" in kinds
