from src.rag.store import RetrievedDoc
from src.rag.retriever import Retriever


class FakeStore:
    def __init__(self, docs):
        self._docs = docs

    def query(self, text, k=6):
        return self._docs[:k]


def test_retriever_falls_back_to_store_order_when_reranker_disabled():
    docs = [
        RetrievedDoc(text="a", source="s1", chunk_id="c1", score=0.9, metadata={}),
        RetrievedDoc(text="b", source="s2", chunk_id="c2", score=0.8, metadata={}),
        RetrievedDoc(text="c", source="s3", chunk_id="c3", score=0.7, metadata={}),
    ]
    store = FakeStore(docs)
    r = Retriever(store=store, top_k=3, rerank_k=2, reranker_model=None)
    r._reranker = "disabled"
    out = r.retrieve("hello")
    assert [d.chunk_id for d in out] == ["c1", "c2"]


def test_retriever_returns_empty_for_empty_store():
    r = Retriever(store=FakeStore([]), top_k=3, rerank_k=2, reranker_model=None)
    r._reranker = "disabled"
    assert r.retrieve("anything") == []
