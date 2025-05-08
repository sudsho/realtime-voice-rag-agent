"""chroma vector store wrapper."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


@dataclass
class Document:
    text: str
    source: str
    chunk_id: str
    metadata: dict


@dataclass
class RetrievedDoc:
    text: str
    source: str
    chunk_id: str
    score: float
    metadata: dict


class ChromaStore:
    def __init__(
        self,
        persist_dir: str = ".chroma",
        collection: str = "support",
        embed_model: str = "text-embedding-3-small",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection
        self.embed_model = embed_model
        self._client = None
        self._coll = None

    def _ensure(self) -> None:
        if self._coll is not None:
            return
        import chromadb  # type: ignore
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction  # type: ignore
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        ef = OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model_name=self.embed_model,
        )
        self._coll = self._client.get_or_create_collection(
            self.collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, docs: Sequence[Document]) -> int:
        if not docs:
            return 0
        self._ensure()
        ids = [d.chunk_id for d in docs]
        texts = [d.text for d in docs]
        metas = [{"source": d.source, **d.metadata} for d in docs]
        self._coll.upsert(ids=ids, documents=texts, metadatas=metas)
        return len(docs)

    def query(self, text: str, k: int = 6) -> List[RetrievedDoc]:
        self._ensure()
        res = self._coll.query(query_texts=[text], n_results=k)
        out: List[RetrievedDoc] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            score = 1.0 - float(dist)  # cosine distance -> similarity
            out.append(RetrievedDoc(
                text=doc,
                source=(meta or {}).get("source", "unknown"),
                chunk_id=cid,
                score=score,
                metadata=meta or {},
            ))
        return out

    def count(self) -> int:
        self._ensure()
        return int(self._coll.count())
