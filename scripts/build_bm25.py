"""scan a KB folder and persist a BM25 index alongside the chroma collection.

we cannot pull from chroma directly because chroma stores embeddings, not the
raw text. so we walk the source dir again here.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha1
from pathlib import Path

from src.rag.bm25 import Bm25Index
from src.rag.chunker import chunk_text


def build(src: str, out_path: str) -> int:
    idx = Bm25Index()
    root = Path(src)
    for fp in sorted(p for p in root.rglob("*") if p.suffix.lower() in {".md", ".txt"}):
        text = fp.read_text(encoding="utf-8", errors="ignore")
        rel = str(fp.relative_to(root))
        for c in chunk_text(text):
            cid = f"{Path(rel).stem}-{c.order:03d}-{sha1((rel + str(c.order)).encode()).hexdigest()[:8]}"
            idx.add(cid, c.text)
    idx.finalize()
    payload = {
        "k1": idx.k1,
        "b": idx.b,
        "avgdl": idx.avgdl,
        "doc_text": idx.doc_text,
        "doc_len": idx.doc_len,
        "df": dict(idx.df),
        "docs": idx.docs,
    }
    Path(out_path).write_text(json.dumps(payload), encoding="utf-8")
    return len(idx.docs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=".chroma/bm25.json")
    args = ap.parse_args(argv)
    n = build(args.src, args.out)
    print(f"indexed {n} docs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
