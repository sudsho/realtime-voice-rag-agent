"""accumulate LLM tokens and emit speakable units to the TTS.

we don't wait for the full response. as soon as a clause boundary lands
(period, question, exclamation, semicolon followed by space) we hand that
piece to TTS so the user starts hearing audio while the LLM still types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


_BOUNDARY = re.compile(r"([.!?;,])\s+")
_FAST_FLUSH_CHARS = 6    # for the very first piece, flush early to get audio out the door
_MIN_FLUSH_CHARS = 24
_MAX_PIECE_CHARS = 200   # avoid handing TTS a paragraph; force a break


@dataclass
class SentenceBuffer:
    pending: str = ""
    _emitted: int = 0

    def _next_piece(self, min_len: int) -> str | None:
        """return the shortest leading clause that is at least ``min_len`` chars.

        scans clause boundaries left to right and takes the first one whose
        piece clears the threshold. a short leading clause no longer blocks the
        buffer: we keep looking for a later boundary that has enough text.
        """
        for m in _BOUNDARY.finditer(self.pending):
            end = m.end()
            piece = self.pending[:end].strip()
            if piece and len(piece) >= min_len:
                self.pending = self.pending[end:]
                return piece
        return None

    def push(self, token: str) -> List[str]:
        self.pending += token
        out: List[str] = []
        while True:
            # for the FIRST piece, flush early to keep TTFB low
            min_len = _FAST_FLUSH_CHARS if self._emitted == 0 else _MIN_FLUSH_CHARS
            piece = self._next_piece(min_len)
            if piece is not None:
                out.append(piece)
                self._emitted += 1
                continue
            # no usable boundary, but a paragraph-length blob - force a break
            if len(self.pending) > _MAX_PIECE_CHARS:
                # break at last whitespace
                cut = self.pending.rfind(" ", 0, _MAX_PIECE_CHARS)
                if cut <= 0:
                    cut = _MAX_PIECE_CHARS
                piece = self.pending[:cut].strip()
                self.pending = self.pending[cut:].lstrip()
                if piece:
                    out.append(piece)
                    self._emitted += 1
                continue
            break
        return out

    def flush(self) -> List[str]:
        rest = self.pending.strip()
        self.pending = ""
        return [rest] if rest else []
