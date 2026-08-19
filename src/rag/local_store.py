"""offline in-memory vector store.

drop-in replacement for :class:`src.rag.store.ChromaStore` that needs no
chromadb server, no OpenAI embedding key, and no network. documents are
embedded with a deterministic feature-hashing embedding (bag of hashed,
plural-stemmed tokens with signed buckets) so retrieval is reproducible across
processes without downloading a model. good enough to rank a handful of KB
articles by keyword overlap for a smoke run.

if ``sentence-transformers`` happens to be installed it is NOT used here; the
hashing embedding keeps the offline path dependency-free (numpy only). the
store exposes the same ``upsert`` / ``query`` / ``count`` surface the retriever
expects, and can lazily ingest a folder of KB files on first use.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.rag.bm25 import tokenize
from src.rag.chunker import chunk_text
from src.rag.store import Document, RetrievedDoc


def _bucket(token: str, dim: int) -> tuple[int, float]:
    """deterministic (index, sign) for a token via md5 feature hashing.

    md5 is used instead of the builtin ``hash`` because ``hash`` is salted per
    process, which would make embeddings non-reproducible across runs.
    """
    h = hashlib.md5(token.encode("utf-8")).digest()
    idx = int.from_bytes(h[:4], "little") % dim
    sign = 1.0 if (h[4] & 1) else -1.0
    return idx, sign


class LocalVectorStore:
    def __init__(
        self,
        kb_dir: str | None = None,
        collection: str = "support",
        dim: int = 1024,
        max_tokens: int = 350,
    ):
        self.kb_dir = kb_dir
        self.collection_name = collection
        self.dim = dim
        self.max_tokens = max_tokens
        self._docs: list[RetrievedDoc] = []
        self._vecs: np.ndarray | None = None
        self._ingested = False

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            idx, sign = _bucket(tok, self.dim)
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def _rebuild_matrix(self) -> None:
        if self._docs:
            self._vecs = np.vstack([self._embed(d.text) for d in self._docs])
        else:
            self._vecs = np.zeros((0, self.dim), dtype=np.float32)

    def upsert(self, docs: Sequence[Document]) -> int:
        if not docs:
            return 0
        by_id = {d.chunk_id: i for i, d in enumerate(self._docs)}
        for d in docs:
            rd = RetrievedDoc(
                text=d.text,
                source=d.source,
                chunk_id=d.chunk_id,
                score=0.0,
                metadata={"source": d.source, **d.metadata},
            )
            if d.chunk_id in by_id:
                self._docs[by_id[d.chunk_id]] = rd
            else:
                by_id[d.chunk_id] = len(self._docs)
                self._docs.append(rd)
        self._rebuild_matrix()
        return len(docs)

    def ingest_dir(self, root: str) -> int:
        root_path = Path(root)
        allowed = {".md", ".txt", ".markdown"}
        files = sorted(
            p for p in root_path.rglob("*")
            if p.is_file() and p.suffix.lower() in allowed
        )
        docs: list[Document] = []
        for fp in files:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            rel = str(fp.relative_to(root_path))
            for c in chunk_text(text, max_tokens=self.max_tokens):
                h = hashlib.sha1(
                    f"{rel}|{c.order}|{c.text[:80]}".encode()
                ).hexdigest()[:16]
                docs.append(Document(
                    text=c.text,
                    source=rel,
                    chunk_id=f"{fp.stem}-{c.order:03d}-{h}",
                    metadata={"order": c.order, "title": fp.stem},
                ))
        return self.upsert(docs)

    def _ensure(self) -> None:
        if self._ingested:
            return
        self._ingested = True
        if self.kb_dir and Path(self.kb_dir).exists() and not self._docs:
            self.ingest_dir(self.kb_dir)

    def query(self, text: str, k: int = 6) -> list[RetrievedDoc]:
        self._ensure()
        if self._vecs is None or self._vecs.shape[0] == 0:
            return []
        q = self._embed(text)
        sims = self._vecs @ q  # cosine (all rows are unit-normalized)
        order = np.argsort(-sims)[:k]
        out: list[RetrievedDoc] = []
        for i in order:
            d = self._docs[int(i)]
            out.append(RetrievedDoc(
                text=d.text,
                source=d.source,
                chunk_id=d.chunk_id,
                score=float(sims[int(i)]),
                metadata=d.metadata,
            ))
        return out

    def count(self) -> int:
        self._ensure()
        return len(self._docs)
