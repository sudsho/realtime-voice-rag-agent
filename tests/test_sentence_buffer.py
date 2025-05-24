from src.agent.sentence_buffer import SentenceBuffer


def test_emits_first_clause_early():
    sb = SentenceBuffer()
    out = sb.push("Hi there. ")
    assert out
    assert out[0].endswith(".")


def test_holds_short_fragments_until_long_enough():
    sb = SentenceBuffer()
    sb._emitted = 1  # simulate after the first clause was already flushed
    short = sb.push("Yes. ")
    # too short for the post-first threshold (24); should hold
    assert short == []
    more = sb.push("Here is the longer second sentence to flush. ")
    assert more
    assert "longer" in more[0]


def test_flush_returns_remainder():
    sb = SentenceBuffer()
    sb.push("trailing fragment without punctuation")
    rest = sb.flush()
    assert rest and "fragment" in rest[0]
    assert sb.flush() == []


def test_force_break_on_long_runaway():
    sb = SentenceBuffer()
    sb._emitted = 1
    big = "word " * 80  # 400 chars, no punctuation
    out = sb.push(big)
    # should force at least one break instead of buffering forever
    assert out
