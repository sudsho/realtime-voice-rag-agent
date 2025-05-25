import time

from src.metrics import TurnMetrics


def test_marks_record_offsets():
    m = TurnMetrics(turn_id="t1")
    m.mark("a")
    time.sleep(0.005)
    m.mark("b")
    a = m.get_ms("a")
    b = m.get_ms("b")
    assert a is not None and b is not None
    assert b > a
    assert m.get_ms("missing") is None


def test_stage_returns_difference():
    m = TurnMetrics(turn_id="t2")
    m.mark("start")
    time.sleep(0.002)
    m.mark("end")
    delta = m.stage("start", "end")
    assert delta is not None and delta >= 0


def test_as_dict_keys_have_ms_suffix():
    m = TurnMetrics(turn_id="t3")
    m.mark("foo")
    d = m.as_dict()
    assert "foo_ms" in d
