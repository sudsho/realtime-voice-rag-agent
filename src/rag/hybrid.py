"""hybrid retrieval: dense (chroma) + sparse (BM25) merged via RRF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.rag.bm25 import Bm25Index
from src.rag.store import ChromaStore, RetrievedDoc


@dataclass
class HybridConfig:
    top_k: int = 10
    rrf_k: int = 60
    final_k: int = 6


class HybridRetriever:
    def __init__(self, store: ChromaStore, bm25: Bm25Index, cfg: HybridConfig | None = None):
        self.store = store
        self.bm25 = bm25
        self.cfg = cfg or HybridConfig()

    def search(self, query: str) -> List[RetrievedDoc]:
        dense = self.store.query(query, k=self.cfg.top_k)
        sparse = self.bm25.search(query, k=self.cfg.top_k)

        rrf: Dict[str, float] = {}
        meta: Dict[str, RetrievedDoc] = {}
        for rank, d in enumerate(dense):
            rrf[d.chunk_id] = rrf.get(d.chunk_id, 0.0) + 1.0 / (self.cfg.rrf_k + rank + 1)
            meta[d.chunk_id] = d
        for rank, h in enumerate(sparse):
            rrf[h.doc_id] = rrf.get(h.doc_id, 0.0) + 1.0 / (self.cfg.rrf_k + rank + 1)
            if h.doc_id not in meta:
                meta[h.doc_id] = RetrievedDoc(
                    text=h.text,
                    source="bm25",
                    chunk_id=h.doc_id,
                    score=h.score,
                    metadata={},
                )

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        out: List[RetrievedDoc] = []
        for cid, s in ranked[: self.cfg.final_k]:
            doc = meta[cid]
            doc.score = s
            out.append(doc)
        return out
