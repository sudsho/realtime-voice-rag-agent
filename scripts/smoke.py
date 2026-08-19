"""offline end-to-end smoke for the voice RAG agent.

Runs the whole STT -> RAG -> LLM -> TTS turn with NO API keys, NO model
downloads, NO GPU, and NO cloud, using the offline config
(`configs/offline.yaml`):

  * STT  -> mock ASR that returns a canned transcript from the buffered audio
  * RAG  -> in-memory hashing-embedding store over the bundled KB articles
  * LLM  -> local extractive generator over the retrieved context
  * TTS  -> mock synthesizer that emits synthetic PCM16 audio bytes

Two smokes run:

  1. the pipeline directly over one synthetic-audio turn (STT->RAG->LLM->TTS)
  2. the WebSocket gateway in-process, driving a scripted start/audio/stop turn

Exit code is non-zero if any assertion fails, so `make smoke` is a real gate.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.pipeline import VoicePipeline  # noqa: E402
from src.audio import TARGET_SR, float32_to_pcm16  # noqa: E402
from src.config import load_config  # noqa: E402
from src.ws.server import VoiceWsHandler  # noqa: E402
from src.ws.session import Session  # noqa: E402

CONFIG_PATH = str(ROOT / "configs" / "offline.yaml")


def synth_audio(seconds: float = 1.2, sr: int = TARGET_SR, freq: float = 140.0) -> np.ndarray:
    """a synthetic voiced-ish tone standing in for a recorded utterance."""
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (0.2 * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


class FakeWebSocket:
    """captures server -> client sends, and (optionally) serves scripted recvs."""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.sent_text = []
        self.sent_bytes = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def receive(self):
        if not self.script:
            return {"type": "websocket.disconnect"}
        return self.script.pop(0)

    async def send_text(self, s: str):
        self.sent_text.append(s)

    async def send_bytes(self, b: bytes):
        self.sent_bytes.append(b)

    def texts(self):
        return [json.loads(t) for t in self.sent_text]


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


async def smoke_pipeline(cfg: dict) -> None:
    print("=" * 70)
    print("SMOKE 1: full STT -> RAG -> LLM -> TTS pipeline over one turn")
    print("=" * 70)

    pipeline = VoicePipeline(cfg)

    # show that the offline backends are the ones actually wired in
    print(f"  STT backend : {type(pipeline.stt).__name__}")
    print(f"  RAG store   : {type(pipeline.retriever.store).__name__} "
          f"({pipeline.retriever.store.count()} KB chunks)")
    print(f"  LLM backend : {type(pipeline.llm).__name__}")
    print(f"  TTS backend : {type(pipeline.tts).__name__} @ {pipeline.tts.sample_rate} Hz")

    # a synthetic utterance the mock STT will "transcribe" to the canned question
    session = Session()
    session.client_sr = TARGET_SR
    session.append_speech(synth_audio())
    print(f"\n  fed {session.speech_buffer.size} synthetic audio samples "
          f"(~{session.speech_buffer.size / TARGET_SR:.2f}s)")

    ws = FakeWebSocket()
    await pipeline.run_turn(session, ws)

    msgs = ws.texts()
    types = [m["type"] for m in msgs]

    transcript = next((m["text"] for m in msgs if m["type"] == "transcript"), None)
    stages = [m["stage"] for m in msgs if m["type"] == "agent_status"]
    answer = "".join(m["text"] for m in msgs if m["type"] == "answer_delta")
    done = next((m for m in msgs if m["type"] == "answer_done"), None)

    docs = pipeline.retriever.retrieve(transcript or "")
    print(f"\n  transcript  : {transcript!r}")
    print(f"  stages      : {stages}")
    print("  retrieved   :")
    for d in docs:
        print(f"      - {d.source}  (score={d.score:.3f})")
    print(f"  answer      : {answer!r}")
    print(f"  tts frames  : {len(ws.sent_bytes)} "
          f"({sum(len(b) for b in ws.sent_bytes)} PCM16 bytes)")
    print(f"  metrics(ms) : {done.get('metrics') if done else None}")

    _require(transcript, "no transcript emitted")
    _require("transcribing" in stages and "retrieving" in stages and "generating" in stages,
             f"missing pipeline stages: {stages}")
    _require(len(docs) > 0, "RAG returned no documents")
    _require(any("billing" in d.source for d in docs), "expected billing article in retrieval")
    _require("14 days" in answer, f"extractive answer missing expected fact: {answer!r}")
    _require(len(ws.sent_bytes) > 0 and sum(len(b) for b in ws.sent_bytes) > 0,
             "no TTS audio bytes produced")
    _require(done is not None and "answer_done" in types, "no answer_done emitted")
    print("\n  SMOKE 1 PASSED")


async def smoke_ws_gateway(cfg: dict) -> None:
    print()
    print("=" * 70)
    print("SMOKE 2: WebSocket gateway in-process (start -> audio -> stop)")
    print("=" * 70)

    pipeline = VoicePipeline(cfg)
    handler = VoiceWsHandler(pipeline)

    blob = float32_to_pcm16(synth_audio())
    script = [
        {"text": json.dumps({"type": "start", "client_sr": TARGET_SR})},
        {"bytes": blob},
        {"text": json.dumps({"type": "stop"})},
    ]
    ws = FakeWebSocket(script)

    # run the handler; the 'stop' schedules a turn task that may outlive the
    # receive loop, so wait until answer_done lands (or time out).
    handler_task = asyncio.create_task(handler(ws))
    for _ in range(500):
        if any(json.loads(t).get("type") == "answer_done" for t in ws.sent_text):
            break
        await asyncio.sleep(0.01)
    await handler_task
    # let the detached turn task finish flushing if needed
    for _ in range(200):
        if any(json.loads(t).get("type") == "answer_done" for t in ws.sent_text):
            break
        await asyncio.sleep(0.01)

    types = [json.loads(t).get("type") for t in ws.sent_text]
    answer = "".join(json.loads(t).get("text", "")
                     for t in ws.sent_text if json.loads(t).get("type") == "answer_delta")
    print(f"  server msg types : {types}")
    print(f"  answer           : {answer!r}")
    print(f"  tts frames       : {len(ws.sent_bytes)} "
          f"({sum(len(b) for b in ws.sent_bytes)} bytes)")

    _require(ws.accepted, "ws was never accepted")
    _require("ready" in types, "gateway never sent 'ready'")
    _require("transcript" in types, "gateway never sent a transcript")
    _require("answer_done" in types, "gateway never completed the turn")
    _require(len(ws.sent_bytes) > 0, "gateway streamed no audio")
    print("\n  SMOKE 2 PASSED")


async def main() -> int:
    cfg = load_config(CONFIG_PATH)
    await smoke_pipeline(cfg)
    await smoke_ws_gateway(cfg)
    print()
    print("=" * 70)
    print("ALL OFFLINE SMOKES PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
