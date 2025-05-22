from src.rag.chunker import chunk_text


def test_short_text_one_chunk():
    text = "Hello world. This is short."
    chunks = chunk_text(text, max_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].text.strip() == text.strip()


def test_paragraph_split():
    parts = ["A " * 100, "B " * 100, "C " * 100]
    text = "\n\n".join(parts)
    chunks = chunk_text(text, max_tokens=80, overlap_tokens=10)
    assert len(chunks) >= 2
    # chunks are ordered
    orders = [c.order for c in chunks]
    assert orders == sorted(orders)


def test_overlap_keeps_some_tail():
    text = ("alpha " * 50) + "\n\n" + ("beta " * 50) + "\n\n" + ("gamma " * 50)
    chunks = chunk_text(text, max_tokens=60, overlap_tokens=20)
    assert len(chunks) >= 2
    # second chunk must repeat some of the previous tail (overlap)
    assert any("alpha" in c.text or "beta" in c.text for c in chunks[1:])
