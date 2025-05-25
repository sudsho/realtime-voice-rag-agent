import numpy as np

from src.audio import (
    FrameChunker, float32_to_pcm16, pcm16_to_float32, resample, to_mono, TARGET_SR,
)


def test_pcm16_roundtrip():
    src = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)
    enc = float32_to_pcm16(src)
    dec = pcm16_to_float32(enc)
    assert dec.shape == src.shape
    np.testing.assert_allclose(dec, src, atol=1e-3)


def test_to_mono_passthrough_for_one_channel():
    arr = np.arange(8, dtype=np.float32)
    out = to_mono(arr, 1)
    assert out is arr or np.array_equal(out, arr)


def test_to_mono_averages_stereo():
    # interleaved L=1, R=3 -> mean=2 (and back)
    arr = np.array([1.0, 3.0, 1.0, 3.0], dtype=np.float32)
    out = to_mono(arr, 2)
    np.testing.assert_array_equal(out, np.array([2.0, 2.0], dtype=np.float32))


def test_to_mono_handles_odd_count():
    # odd count drops the trailing sample rather than crashing
    arr = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    out = to_mono(arr, 2)
    np.testing.assert_array_equal(out, np.array([2.0], dtype=np.float32))


def test_resample_passthrough_when_same_rate():
    arr = np.arange(10, dtype=np.float32)
    out = resample(arr, TARGET_SR, TARGET_SR)
    assert out is arr


def test_resample_changes_length():
    arr = np.linspace(0, 1, 1000, dtype=np.float32)
    out = resample(arr, 48000, 16000)
    # roughly a third the length; loose bounds because of integer rounding
    expected = round(1000 * 16000 / 48000)
    assert abs(out.size - expected) <= 2


def test_chunker_emits_fixed_size_frames():
    ch = FrameChunker(sample_rate=16000, frame_ms=30)
    samples = np.zeros(int(16000 * 0.1), dtype=np.float32)  # 100ms
    frames = ch.feed(samples)
    # 100ms / 30ms = 3 frames, 10ms tail
    assert len(frames) == 3
    for f in frames:
        assert f.pcm.size == 480
        assert f.duration_ms == 30


def test_chunker_flush_pads_tail():
    ch = FrameChunker(sample_rate=16000, frame_ms=30)
    ch.feed(np.zeros(100, dtype=np.float32))
    frame = ch.flush()
    assert frame is not None
    assert frame.pcm.size == 480
