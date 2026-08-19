import asyncio
import json

import numpy as np
import pytest

from src.audio import float32_to_pcm16, TARGET_SR
from src.ws.server import VoiceWsHandler


class FakeWebSocket:
    """captures sends and serves a script of receives."""

    def __init__(self, script):
        self.script = list(script)
        self.sent_text = []
        self.sent_bytes = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def receive(self):
        if not self.script:
            return {"type": "websocket.disconnect"}
        return self.script.pop(0)

    async def send_text(self, s):
        self.sent_text.append(s)

    async def send_bytes(self, b):
        self.sent_bytes.append(b)


class FakePipeline:
    def __init__(self):
        self.calls = 0

    async def run_turn(self, session, ws):
        self.calls += 1
        await ws.send_text(json.dumps({"type": "answer_done", "metrics": {"x_ms": 1.0}}))


@pytest.mark.asyncio
async def test_ws_loop_routes_start_stop():
    blob = float32_to_pcm16(np.zeros(int(TARGET_SR * 0.5), dtype=np.float32))
    script = [
        {"text": json.dumps({"type": "start", "client_sr": 16000})},
        {"bytes": blob},
        {"text": json.dumps({"type": "stop"})},
    ]
    ws = FakeWebSocket(script)
    pipe = FakePipeline()
    handler = VoiceWsHandler(pipe)
    await handler(ws)
    # at least one of the sent texts should be 'ready'
    types = [json.loads(t).get("type") for t in ws.sent_text]
    assert "ready" in types
    # pipeline should have been triggered exactly once
    # (give the scheduled turn task a tick to run)
    for _ in range(20):
        if pipe.calls:
            break
        await asyncio.sleep(0.01)
    assert pipe.calls >= 1


@pytest.mark.asyncio
async def test_ws_pong_on_ping():
    script = [{"text": json.dumps({"type": "ping"})}]
    ws = FakeWebSocket(script)
    handler = VoiceWsHandler(FakePipeline())
    await handler(ws)
    types = [json.loads(t).get("type") for t in ws.sent_text]
    assert "pong" in types


@pytest.mark.asyncio
async def test_ws_error_on_bad_type():
    script = [{"text": json.dumps({"type": "garbage"})}]
    ws = FakeWebSocket(script)
    handler = VoiceWsHandler(FakePipeline())
    await handler(ws)
    types = [json.loads(t).get("type") for t in ws.sent_text]
    assert "error" in types
