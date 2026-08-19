"""tests for the offline backends (mock STT, local store, extractive LLM,
mock TTS) and the full offline pipeline turn.

these exercise the same code path `scripts/smoke.py` runs, with no keys, no
downloads, and no GPU.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.agent.local_llm import ExtractiveLlm
from src.agent.pipeline import VoicePipeline
from src.config import load_config
from src.rag.factory import make_store
from src.rag.local_store import LocalVectorStore
from src.stt.factory import make_stt
from src.stt.mock_stt import MockStt
from src.tts.factory import make_tts
from src.tts.mock_tts import MockTts

ROOT = Path(__file__).resolve().parents[1]
OFFLINE_CFG = str(ROOT / "configs" / "offline.yaml")
KB_DIR = str(ROOT / "data" / "sample_kb")


def _tone(seconds=1.0, sr=16000):
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (0.2 * np.sin(2 * math.pi * 140 * t)).astype(np.float32)


def test_mock_stt_returns_canned_transcript():
    stt = MockStt(transcript="hello world")
    out = list(stt.transcribe(_tone(0.5), sample_rate=16000))
    assert len(out) == 1
    assert out[0].text == "hello world"
    assert out[0].end_ms > 0  # duration derived from the audio length


def test_mock_tts_emits_synthetic_pcm_bytes():
    tts = MockTts()
    frames = []

    async def _collect():
        async for b in tts.synth_stream("some text to speak"):
            frames.append(b)

    import asyncio

    asyncio.run(_collect())
    assert frames
    total = b"".join(frames)
    assert len(total) > 0 and len(total) % 2 == 0  # PCM16
    assert total != b"\x00" * len(total)  # not silence


def test_local_store_ingests_and_ranks_kb():
    store = LocalVectorStore(kb_dir=KB_DIR)
    assert store.count() == 7  # one chunk per KB article
    hits = store.query("how long do I have to get a refund?", k=3)
    assert hits
    assert any("billing" in h.source for h in hits)
    # scores are cosine similarities in [-1, 1], descending
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_factories_select_offline_backends():
    cfg = load_config(OFFLINE_CFG)
    assert type(make_stt(cfg)).__name__ == "MockStt"
    assert type(make_store(cfg)).__name__ == "LocalVectorStore"
    assert type(make_tts(cfg)).__name__ == "MockTts"


def test_extractive_llm_grounds_answer_in_context():
    store = LocalVectorStore(kb_dir=KB_DIR)
    docs = store.query("how long do I have to get a refund?", k=3)
    from src.agent.prompt import build_messages

    messages = build_messages("How long do I have to get a refund?", docs)
    llm = ExtractiveLlm(max_sentences=2)

    async def _run():
        return [d async for d in llm.stream(messages)]

    import asyncio

    deltas = asyncio.run(_run())
    answer = "".join(d.text for d in deltas)
    assert deltas[-1].finished is True
    assert "14 days" in answer  # the grounded refund fact


def test_extractive_llm_escalates_without_context():
    llm = ExtractiveLlm()
    messages = [{"role": "user", "content": "User question: what is the meaning of life?"}]

    async def _run():
        return "".join([d.text async for d in llm.stream(messages)])

    import asyncio

    answer = asyncio.run(_run())
    assert "escalate" in answer.lower()


@pytest.mark.asyncio
async def test_offline_pipeline_full_turn():
    from src.ws.session import Session

    class FakeWs:
        def __init__(self):
            self.sent_text = []
            self.sent_bytes = []

        async def send_text(self, s):
            self.sent_text.append(s)

        async def send_bytes(self, b):
            self.sent_bytes.append(b)

    cfg = load_config(OFFLINE_CFG)
    pipeline = VoicePipeline(cfg)
    session = Session()
    session.client_sr = 16000
    session.append_speech(_tone(1.2))

    ws = FakeWs()
    await pipeline.run_turn(session, ws)

    types = [json.loads(t)["type"] for t in ws.sent_text]
    answer = "".join(json.loads(t).get("text", "")
                     for t in ws.sent_text if json.loads(t)["type"] == "answer_delta")
    assert "transcript" in types
    assert "answer_done" in types
    assert "14 days" in answer
    assert len(ws.sent_bytes) > 0
