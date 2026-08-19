"""pick the RAG vector store backend from config.

``rag.backend`` selects between chroma (default, needs an OpenAI embedding key)
and the offline local hashing store (no keys, no network, no server).
"""

from __future__ import annotations


def make_store(cfg: dict | None = None):
    cfg = cfg or {}
    rag_cfg = cfg.get("rag", {})
    backend = (rag_cfg.get("backend") or "chroma").lower()
    if backend in ("local", "offline", "memory"):
        from src.rag.local_store import LocalVectorStore

        return LocalVectorStore(
            kb_dir=rag_cfg.get("kb_dir", "data/sample_kb"),
            collection=rag_cfg.get("collection", "support"),
        )
    from src.rag.store import ChromaStore

    return ChromaStore(
        persist_dir=rag_cfg.get("chroma_dir", ".chroma"),
        collection=rag_cfg.get("collection", "support"),
        embed_model=rag_cfg.get("embed_model", "text-embedding-3-small"),
    )
