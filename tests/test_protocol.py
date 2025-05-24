import json

import pytest

from src.ws.protocol import (
    ProtoError, decode_text, encode_text, msg_answer_delta, msg_answer_done,
    msg_error, msg_ready, msg_status, msg_transcript, msg_tts_meta,
)


def test_encode_decode_roundtrip():
    s = encode_text({"type": "ping"})
    assert json.loads(s) == {"type": "ping"}
    assert decode_text(s) == {"type": "ping"}


def test_decode_rejects_non_object():
    with pytest.raises(ProtoError):
        decode_text("[1, 2, 3]")


def test_decode_rejects_missing_type():
    with pytest.raises(ProtoError):
        decode_text('{"foo": 1}')


def test_decode_rejects_bad_json():
    with pytest.raises(ProtoError):
        decode_text("not-json")


def test_helpers_emit_correct_types():
    assert json.loads(msg_ready())["type"] == "ready"
    assert json.loads(msg_transcript("hi", True))["type"] == "transcript"
    assert json.loads(msg_status("generating"))["stage"] == "generating"
    assert json.loads(msg_answer_delta("ab"))["text"] == "ab"
    assert json.loads(msg_tts_meta(24000))["sr"] == 24000
    assert json.loads(msg_answer_done({"x": 1}))["metrics"] == {"x": 1}
    assert json.loads(msg_error("c", "m"))["code"] == "c"
