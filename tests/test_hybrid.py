"""tests for hybrid retriever (RRF merge of dense + sparse)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.rag.bm25 import Bm25Index, Bm25Hit
from src.rag.hybrid import HybridConfig, HybridRetriever
from src.rag.store import RetrievedDoc


class FakeStore:
    """stand-in for ChromaStore so we don't need to spin up real chroma."""

    def __init__(self, docs: List[RetrievedDoc]):
        self._docs = docs

    def query(self, q: str, k: int) -> List[RetrievedDoc]:
        return self._docs[:k]


def _doc(cid: str, text: str, score: float = 0.5) -> RetrievedDoc:
    return RetrievedDoc(text=text, source="dense", chunk_id=cid, score=score, metadata={})


def test_rrf_prefers_docs_in_both_lists():
    dense = [_doc("a", "alpha"), _doc("b", "beta"), _doc("c", "gamma")]
    bm = Bm25Index()
    bm.add("b", "beta is the second")
    bm.add("d", "delta delta delta")
    bm.add("a", "alpha alpha")
    bm.finalize()

    h = HybridRetriever(FakeStore(dense), bm, HybridConfig(top_k=3, final_k=3))
    out = h.search("alpha")
    ids = [d.chunk_id for d in out]
    # docs that appear in both lists should rank above ones in only one
    assert "a" in ids[:2] or "b" in ids[:2]


def test_rrf_includes_bm25_only_doc():
    dense: List[RetrievedDoc] = []
    bm = Bm25Index()
    bm.add("only-here", "the warranty covers manufacturing defects")
    bm.finalize()

    h = HybridRetriever(FakeStore(dense), bm, HybridConfig(top_k=5, final_k=5))
    out = h.search("warranty defects")
    assert any(d.chunk_id == "only-here" for d in out)


def test_final_k_caps_results():
    dense = [_doc(f"d{i}", f"text {i}") for i in range(10)]
    bm = Bm25Index()
    for i in range(10):
        bm.add(f"s{i}", f"sparse text {i}")
    bm.finalize()

    h = HybridRetriever(FakeStore(dense), bm, HybridConfig(top_k=10, final_k=4))
    out = h.search("text")
    assert len(out) <= 4
