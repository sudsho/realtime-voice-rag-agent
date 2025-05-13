"""websocket gateway. handles audio in, transcript/audio out.

the actual STT/RAG/LLM/TTS work happens in src/agent/pipeline.py. this file
is the IO boundary between the browser and the pipeline.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import numpy as np

from fastapi import WebSocket, WebSocketDisconnect

from src.audio import pcm16_to_float32, resample, to_mono, TARGET_SR
from src.config import load_config
from src.logging_utils import get_logger
from src.ws.protocol import (
    decode_text, msg_error, msg_ready, ProtoError,
)
from src.ws.session import Session


log = get_logger(__name__)


class VoiceWsHandler:
    def __init__(self, pipeline):
        # pipeline is duck-typed: must expose run_turn(session, ws) coroutine
        self.pipeline = pipeline
        self.cfg = load_config()
        self._max_bytes = int(self.cfg.get("ws", {}).get("max_message_bytes", 1 << 20))

    async def __call__(self, ws: WebSocket) -> None:
        await ws.accept()
        session = Session()
        await ws.send_text(msg_ready())
        log.info("ws.connected", extra={"session_id": session.session_id})
        try:
            await self._loop(ws, session)
        except WebSocketDisconnect:
            log.info("ws.disconnected", extra={"session_id": session.session_id})
        except Exception as e:  # noqa: BLE001
            log.exception("ws.unhandled", extra={"session_id": session.session_id})
            try:
                await ws.send_text(msg_error("internal", str(e)))
            except Exception:
                pass

    async def _loop(self, ws: WebSocket, session: Session) -> None:
        running_turn: Optional[asyncio.Task] = None
        while True:
            recv = await ws.receive()
            kind = recv.get("type")
            if kind == "websocket.disconnect":
                break
            if "text" in recv and recv["text"] is not None:
                payload = self._safe_decode(recv["text"])
                if payload is None:
                    continue
                msg_type = payload.get("type")
                if msg_type == "start":
                    session.client_sr = int(payload.get("client_sr", 48000))
                    session.reset_speech()
                elif msg_type == "audio_meta":
                    session.client_sr = int(payload.get("sr", session.client_sr))
                elif msg_type == "stop":
                    if running_turn and not running_turn.done():
                        continue
                    running_turn = asyncio.create_task(
                        self.pipeline.run_turn(session, ws)
                    )
                elif msg_type == "interrupt":
                    session.cancel()
                    if running_turn and not running_turn.done():
                        running_turn.cancel()
                elif msg_type == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
                else:
                    await ws.send_text(msg_error("bad_type", f"unknown: {msg_type}"))
            elif "bytes" in recv and recv["bytes"] is not None:
                self._ingest_audio(session, recv["bytes"])
            else:
                continue

    def _safe_decode(self, text: str) -> Optional[dict[str, Any]]:
        try:
            return decode_text(text)
        except ProtoError as e:
            log.warning("proto.bad_msg", extra={"err": str(e)})
            return None

    def _ingest_audio(self, session: Session, blob: bytes) -> None:
        if not blob:
            return
        # client sends int16 mono in source SR; some browsers send stereo
        pcm = pcm16_to_float32(blob)
        pcm = to_mono(pcm, channels=1)
        if session.client_sr != TARGET_SR:
            pcm = resample(pcm, session.client_sr, TARGET_SR)
        session.append_speech(pcm)
