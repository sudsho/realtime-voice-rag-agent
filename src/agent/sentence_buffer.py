"""accumulate LLM tokens and emit speakable units to the TTS.

we don't wait for the full response. as soon as a clause boundary lands
(period, question, exclamation, semicolon followed by space) we hand that
piece to TTS so the user starts hearing audio while the LLM still types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


_BOUNDARY = re.compile(r"([.!?;])\s+")
_MIN_FLUSH_CHARS = 24


@dataclass
class SentenceBuffer:
    pending: str = ""

    def push(self, token: str) -> List[str]:
        self.pending += token
        out: List[str] = []
        # find clause boundaries and emit each completed piece
        while True:
            m = _BOUNDARY.search(self.pending)
            if not m:
                break
            end = m.end()
            piece = self.pending[:end].strip()
            if piece and len(piece) >= _MIN_FLUSH_CHARS:
                out.append(piece)
                self.pending = self.pending[end:]
            else:
                # short fragment, glue with the next clause
                break
        return out

    def flush(self) -> List[str]:
        rest = self.pending.strip()
        self.pending = ""
        return [rest] if rest else []
