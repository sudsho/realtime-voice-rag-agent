"""websocket message protocol.

binary frames are raw audio (pcm16, 16kHz mono from client; 24kHz from server).
text frames are JSON control messages.

client -> server:
  {"type": "start", "session_id": "...", "client_sr": 48000}
  {"type": "audio_meta", "format": "pcm16", "sr": 48000, "channels": 1}
  binary: pcm chunk
  {"type": "stop"}
  {"type": "interrupt"}
  {"type": "ping"}

server -> client:
  {"type": "ready"}
  {"type": "transcript", "text": "...", "final": true|false}
  {"type": "agent_status", "stage": "retrieving|generating|speaking"}
  {"type": "answer_delta", "text": "..."}
  {"type": "tts_meta", "sr": 24000}
  binary: tts pcm chunk
  {"type": "answer_done"}
  {"type": "error", "code": "...", "msg": "..."}
  {"type": "pong"}
"""

from __future__ import annotations

import json
from typing import Any, Dict


class ProtoError(Exception):
    pass


def encode_text(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))


def decode_text(s: str) -> Dict[str, Any]:
    try:
        d = json.loads(s)
    except json.JSONDecodeError as e:
        raise ProtoError(f"bad json: {e}") from e
    if not isinstance(d, dict):
        raise ProtoError("payload must be a json object")
    if "type" not in d:
        raise ProtoError("missing 'type'")
    return d


# helpers for frequently-emitted server messages
def msg_ready() -> str:
    return encode_text({"type": "ready"})


def msg_transcript(text: str, final: bool) -> str:
    return encode_text({"type": "transcript", "text": text, "final": final})


def msg_status(stage: str) -> str:
    return encode_text({"type": "agent_status", "stage": stage})


def msg_answer_delta(text: str) -> str:
    return encode_text({"type": "answer_delta", "text": text})


def msg_tts_meta(sr: int) -> str:
    return encode_text({"type": "tts_meta", "sr": sr})


def msg_answer_done(metrics: Dict[str, Any] | None = None) -> str:
    payload: Dict[str, Any] = {"type": "answer_done"}
    if metrics is not None:
        payload["metrics"] = metrics
    return encode_text(payload)


def msg_error(code: str, msg: str) -> str:
    return encode_text({"type": "error", "code": code, "msg": msg})
