"""query-side retrieval. chroma top-k then a small cross-encoder rerank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.rag.store import ChromaStore, RetrievedDoc


@dataclass
class Citation:
    source: str
    chunk_id: str
    score: float


class Retriever:
    def __init__(
        self,
        store: ChromaStore,
        top_k: int = 6,
        rerank_k: int = 3,
        reranker_model: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.store = store
        self.top_k = top_k
        self.rerank_k = rerank_k
        self.reranker_model = reranker_model
        self._reranker = None

    def _ensure_reranker(self) -> None:
        if self._reranker is not None or not self.reranker_model:
            return
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._reranker = CrossEncoder(self.reranker_model)
        except Exception:
            # rerank is optional; fall back to chroma-only ordering
            self._reranker = "disabled"

    def retrieve(self, query: str) -> List[RetrievedDoc]:
        candidates = self.store.query(query, k=self.top_k)
        if not candidates:
            return []
        self._ensure_reranker()
        if self._reranker == "disabled" or self._reranker is None:
            return candidates[: self.rerank_k]
        pairs = [(query, c.text) for c in candidates]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
        out: List[RetrievedDoc] = []
        for c, s in ranked[: self.rerank_k]:
            c.score = float(s)
            out.append(c)
        return out

    @staticmethod
    def to_citations(docs: List[RetrievedDoc]) -> List[Citation]:
        return [Citation(source=d.source, chunk_id=d.chunk_id, score=d.score) for d in docs]
