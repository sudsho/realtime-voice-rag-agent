"""text chunking. token-aware-ish, but cheap; we don't import tiktoken here.

splits on paragraph boundaries first, then on sentence boundaries if a
paragraph is too long. uses an approximate char->token ratio of 4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List


_PARA_RE = re.compile(r"\n\s*\n")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    text: str
    order: int


def _split_long_paragraph(p: str, max_chars: int) -> List[str]:
    if len(p) <= max_chars:
        return [p]
    sents = _SENT_RE.split(p)
    out: List[str] = []
    cur = ""
    for s in sents:
        if not cur:
            cur = s
            continue
        if len(cur) + 1 + len(s) <= max_chars:
            cur = cur + " " + s
        else:
            out.append(cur)
            cur = s
    if cur:
        out.append(cur)
    # if even single sentences are too long, hard split
    final: List[str] = []
    for piece in out:
        if len(piece) <= max_chars:
            final.append(piece)
        else:
            for i in range(0, len(piece), max_chars):
                final.append(piece[i : i + max_chars])
    return final


def chunk_text(text: str, max_tokens: int = 350, overlap_tokens: int = 40) -> List[Chunk]:
    max_chars = max_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    paragraphs = [p.strip() for p in _PARA_RE.split(text) if p.strip()]
    pieces: List[str] = []
    for p in paragraphs:
        pieces.extend(_split_long_paragraph(p, max_chars))

    # pack pieces into chunks under the limit, with overlap between chunks
    chunks: List[Chunk] = []
    cur = ""
    order = 0
    for piece in pieces:
        if not cur:
            cur = piece
            continue
        if len(cur) + 2 + len(piece) <= max_chars:
            cur = cur + "\n\n" + piece
        else:
            chunks.append(Chunk(cur, order))
            order += 1
            tail = cur[-overlap_chars:] if overlap_chars else ""
            cur = (tail + "\n\n" + piece).strip()
    if cur:
        chunks.append(Chunk(cur, order))
    return chunks
