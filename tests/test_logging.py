"""tests for structured (JSON) logging formatter."""

from __future__ import annotations

import json
import logging
from io import StringIO

from src.logging_utils import _JsonFormatter


def _make_record(msg: str, **extras):
    rec = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname="t.py",
        lineno=1,
        msg=msg,
        args=None,
        exc_info=None,
    )
    for k, v in extras.items():
        setattr(rec, k, v)
    return rec


def test_basic_fields_present():
    f = _JsonFormatter()
    out = f.format(_make_record("hello"))
    j = json.loads(out)
    assert j["msg"] == "hello"
    assert j["level"] == "INFO"
    assert j["logger"] == "t"
    assert "ts" in j


def test_extras_included():
    f = _JsonFormatter()
    rec = _make_record("event", session_id="abc-123", latency_ms=42)
    j = json.loads(f.format(rec))
    assert j["session_id"] == "abc-123"
    assert j["latency_ms"] == 42


def test_handler_pipes_json_to_stream():
    buf = StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(_JsonFormatter())
    log = logging.getLogger("test_logging_pipe")
    log.handlers = [h]
    log.setLevel(logging.INFO)
    log.info("ws.open", extra={"sid": "s1"})
    line = buf.getvalue().strip()
    j = json.loads(line)
    assert j["msg"] == "ws.open"
    assert j["sid"] == "s1"
