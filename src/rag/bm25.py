"""tiny BM25 implementation over the in-memory text corpus.

we keep this lightweight to avoid pulling rank_bm25 in. the corpus is small
(KB articles, a few hundred chunks) so a python loop is fine.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _stem(tok: str) -> str:
    """very small plural stemmer so 'refund' matches 'refunds'.

    strips a single trailing 's' for words longer than three chars that do
    not end in 'ss' (keeps 'business', 'process' intact). applied to both the
    corpus and the query, so matching stays consistent either way.
    """
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str) -> List[str]:
    return [_stem(t.lower()) for t in _TOKEN_RE.findall(text)]


@dataclass
class Bm25Hit:
    doc_id: str
    score: float
    text: str


class Bm25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: Dict[str, List[str]] = {}
        self.doc_text: Dict[str, str] = {}
        self.doc_len: Dict[str, int] = {}
        self.avgdl: float = 0.0
        self.df: Counter = Counter()

    def add(self, doc_id: str, text: str) -> None:
        toks = tokenize(text)
        self.docs[doc_id] = toks
        self.doc_text[doc_id] = text
        self.doc_len[doc_id] = len(toks)
        for term in set(toks):
            self.df[term] += 1

    def finalize(self) -> None:
        if not self.doc_len:
            self.avgdl = 0.0
            return
        self.avgdl = sum(self.doc_len.values()) / len(self.doc_len)

    def _idf(self, term: str) -> float:
        n = len(self.docs)
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10) -> List[Bm25Hit]:
        if not self.docs:
            return []
        q_tokens = tokenize(query)
        scores: Dict[str, float] = {}
        for term in q_tokens:
            idf = self._idf(term)
            if idf == 0:
                continue
            for doc_id, toks in self.docs.items():
                tf = toks.count(term)
                if tf == 0:
                    continue
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [Bm25Hit(doc_id=d, score=s, text=self.doc_text[d]) for d, s in ranked]
