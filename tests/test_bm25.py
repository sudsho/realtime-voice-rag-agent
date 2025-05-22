from src.rag.bm25 import Bm25Index, tokenize


def test_tokenize_lowercases_and_keeps_alnum():
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]


def test_bm25_finds_relevant_doc():
    idx = Bm25Index()
    idx.add("a", "refunds are processed within five business days")
    idx.add("b", "delivery ships from the warehouse next day")
    idx.add("c", "two factor authentication uses TOTP codes")
    idx.finalize()
    hits = idx.search("how long for a refund", k=2)
    assert hits
    assert hits[0].doc_id == "a"


def test_bm25_returns_empty_on_empty_index():
    idx = Bm25Index()
    idx.finalize()
    assert idx.search("anything") == []
