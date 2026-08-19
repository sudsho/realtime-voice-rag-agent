"""pick the LLM backend from config.

``llm.provider`` selects between the streaming OpenAI client (default) and the
offline extractive generator (no key, no network).
"""

from __future__ import annotations


def make_llm(cfg: dict | None = None):
    cfg = cfg or {}
    llm_cfg = cfg.get("llm", {})
    provider = (llm_cfg.get("provider") or "openai").lower()
    if provider in ("local", "extractive", "offline", "mock"):
        from src.agent.local_llm import ExtractiveLlm

        return ExtractiveLlm(max_sentences=int(llm_cfg.get("max_sentences", 2)))
    from src.agent.llm import StreamingLlm

    return StreamingLlm(
        model=llm_cfg.get("model", "gpt-4o-mini"),
        temperature=float(llm_cfg.get("temperature", 0.1)),
        max_tokens=int(llm_cfg.get("max_tokens", 256)),
    )
