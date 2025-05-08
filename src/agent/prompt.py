"""prompt assembly for the support RAG agent."""

from __future__ import annotations

from typing import List, Sequence

from src.rag.store import RetrievedDoc


DEFAULT_SYSTEM = (
    "You are a customer support agent. Answer using only the provided context. "
    "Keep answers under three sentences. If the context does not cover the "
    "question, say so plainly and offer to escalate."
)


def render_context(docs: Sequence[RetrievedDoc]) -> str:
    if not docs:
        return "(no relevant articles)"
    blocks: List[str] = []
    for i, d in enumerate(docs, start=1):
        blocks.append(f"[{i}] source={d.source}\n{d.text}")
    return "\n\n".join(blocks)


def build_messages(
    user_text: str,
    docs: Sequence[RetrievedDoc],
    system: str = DEFAULT_SYSTEM,
    history: Sequence[dict] | None = None,
) -> List[dict]:
    msgs: List[dict] = [{"role": "system", "content": system}]
    if history:
        msgs.extend(history)
    ctx = render_context(docs)
    user = (
        f"Context:\n{ctx}\n\n"
        f"User question: {user_text}\n\n"
        "Answer briefly. Cite sources by their bracketed number when useful."
    )
    msgs.append({"role": "user", "content": user})
    return msgs
