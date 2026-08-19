"""offline extractive "LLM".

drop-in replacement for :class:`src.agent.llm.StreamingLlm` that needs no
OpenAI key and no network. instead of generating text, it extracts the
sentences from the retrieved context that best overlap the user's question and
streams them back token by token, so the clause-by-clause TTS handoff and all
the downstream wiring exercise exactly the same code path as the real LLM.

this is deliberately simple (lexical overlap scoring over the provided
context), not a language model. it exists so the pipeline has a deterministic,
key-free generator for the offline smoke.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from src.agent.llm import LlmDelta
from src.rag.bm25 import tokenize

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CTX_BLOCK = re.compile(r"^\[(\d+)\]\s+source=(.*)$")


class ExtractiveLlm:
    def __init__(self, max_sentences: int = 2):
        self.max_sentences = max_sentences

    def _parse_user(self, messages: list[dict]) -> tuple[str, list[tuple[int, str]]]:
        """pull the question and the numbered context blocks out of the prompt.

        the prompt builder (:func:`src.agent.prompt.build_messages`) renders the
        context as ``[i] source=...\\n<text>`` blocks followed by a
        ``User question: ...`` line, so we can recover both here.
        """
        user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user = m.get("content", "")
                break
        question = ""
        mq = re.search(r"User question:\s*(.+)", user)
        if mq:
            question = mq.group(1).strip()

        blocks: list[tuple[int, str]] = []
        cur_idx = None
        cur_lines: list[str] = []
        for line in user.splitlines():
            hb = _CTX_BLOCK.match(line.strip())
            if hb:
                if cur_idx is not None and cur_lines:
                    blocks.append((cur_idx, " ".join(cur_lines).strip()))
                cur_idx = int(hb.group(1))
                cur_lines = []
            elif cur_idx is not None:
                if line.startswith("User question:"):
                    break
                stripped = line.strip()
                # drop markdown heading lines so they don't glue onto the
                # following sentence (no period to split on otherwise)
                if stripped and not stripped.startswith("#"):
                    cur_lines.append(stripped)
        if cur_idx is not None and cur_lines:
            blocks.append((cur_idx, " ".join(cur_lines).strip()))
        return question, blocks

    def _answer(self, messages: list[dict]) -> str:
        question, blocks = self._parse_user(messages)
        if not blocks:
            return "I do not have any relevant context for that, so let me escalate you to a human agent."
        q_terms = set(tokenize(question))

        scored: list[tuple[float, int, str]] = []
        for idx, text in blocks:
            for sent in _SENT_SPLIT.split(text):
                sent = sent.strip()
                if len(sent) < 12:
                    continue
                s_terms = set(tokenize(sent))
                if not s_terms:
                    continue
                overlap = len(q_terms & s_terms)
                if overlap == 0:
                    continue
                score = overlap / (len(q_terms) ** 0.5 + len(s_terms) ** 0.5)
                scored.append((score, idx, sent))

        if not scored:
            return "The provided articles do not cover that, so I will escalate you to a human agent."

        scored.sort(key=lambda x: x[0], reverse=True)
        picked: list[str] = []
        cites: list[int] = []
        seen: set[str] = set()
        for _, idx, sent in scored[: self.max_sentences]:
            if sent in seen:
                continue
            seen.add(sent)
            picked.append(sent)
            if idx not in cites:
                cites.append(idx)
        answer = " ".join(picked)
        if cites:
            answer += " " + "".join(f"[{i}]" for i in sorted(cites))
        return answer

    async def stream(self, messages: list[dict]) -> AsyncIterator[LlmDelta]:
        answer = self._answer(messages)
        # stream token by token (word + trailing space) like a real LLM would,
        # so the sentence buffer / TTS handoff sees a genuine token stream.
        tokens = re.findall(r"\S+\s*", answer)
        for tok in tokens:
            yield LlmDelta(text=tok, finished=False)
        yield LlmDelta(text="", finished=True)
