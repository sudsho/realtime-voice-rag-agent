"""ingest a folder of markdown/text files into a chroma collection."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from src.config import load_config
from src.logging_utils import get_logger, setup_logging
from src.rag.chunker import chunk_text
from src.rag.store import ChromaStore, Document


log = get_logger(__name__)

ALLOWED_EXT = {".md", ".txt", ".markdown"}


def _read_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def _chunk_id(source: str, order: int, body: str) -> str:
    h = hashlib.sha1(f"{source}|{order}|{body[:80]}".encode("utf-8")).hexdigest()[:16]
    return f"{Path(source).stem}-{order:03d}-{h}"


def ingest_dir(root: str, collection: str | None = None) -> int:
    cfg = load_config()
    rag_cfg = cfg.get("rag", {})
    store = ChromaStore(
        persist_dir=rag_cfg.get("chroma_dir", ".chroma"),
        collection=collection or rag_cfg.get("collection", "support"),
        embed_model=rag_cfg.get("embed_model", "text-embedding-3-small"),
    )

    docs: list[Document] = []
    root_path = Path(root)
    files = sorted(p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED_EXT)
    log.info("ingest scanning", extra={"root": str(root_path), "files": len(files)})
    for fp in files:
        text = _read_file(fp)
        chunks = chunk_text(text)
        rel = str(fp.relative_to(root_path)) if fp.is_relative_to(root_path) else str(fp)
        for c in chunks:
            docs.append(Document(
                text=c.text,
                source=rel,
                chunk_id=_chunk_id(rel, c.order, c.text),
                metadata={"order": c.order, "title": fp.stem},
            ))
    n = store.upsert(docs)
    log.info("ingest done", extra={"chunks": n, "files": len(files)})
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ingest KB into chroma")
    parser.add_argument("--src", required=True, help="path to KB folder")
    parser.add_argument("--collection", default=None)
    args = parser.parse_args(argv)
    setup_logging("INFO")
    n = ingest_dir(args.src, args.collection)
    print(f"ingested {n} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
