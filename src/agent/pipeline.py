"""end-to-end turn orchestration.

flow per turn:
  1. user said "stop" -> we have an audio buffer in session.speech_buffer
  2. transcribe (faster-whisper)
  3. retrieve top-k from chroma
  4. stream LLM tokens, every clause -> TTS, TTS bytes -> ws
  5. emit answer_done with metrics

cancellation: session.cancel_event being set, or asyncio.CancelledError, both
short-circuit the rest of the turn.
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import WebSocket

from src.agent.prompt import DEFAULT_SYSTEM, build_messages
from src.agent.sentence_buffer import SentenceBuffer
from src.config import load_config
from src.logging_utils import get_logger
from src.metrics import TurnMetrics
from src.rag.retriever import Retriever
from src.tts.factory import make_tts
from src.ws.protocol import (
    msg_answer_delta,
    msg_answer_done,
    msg_error,
    msg_status,
    msg_transcript,
    msg_tts_meta,
)
from src.ws.session import Session

log = get_logger(__name__)


class VoicePipeline:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self._stt = None
        self._retriever: Retriever | None = None
        self._llm = None
        self._tts = None

    def warm(self) -> None:
        """preload models so the first turn does not pay cold start."""
        _ = self.stt
        _ = self.retriever
        _ = self.llm
        _ = self.tts

    @property
    def stt(self):
        if self._stt is None:
            from src.stt.factory import make_stt
            self._stt = make_stt(self.cfg)
        return self._stt

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            from src.rag.factory import make_store
            rag_cfg = self.cfg.get("rag", {})
            store = make_store(self.cfg)
            self._retriever = Retriever(
                store=store,
                top_k=int(rag_cfg.get("top_k", 6)),
                rerank_k=int(rag_cfg.get("rerank_k", 3)),
                reranker_model=rag_cfg.get(
                    "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
                ),
            )
        return self._retriever

    @property
    def llm(self):
        if self._llm is None:
            from src.agent.llm_factory import make_llm
            self._llm = make_llm(self.cfg)
        return self._llm

    @property
    def tts(self):
        if self._tts is None:
            self._tts = make_tts(self.cfg)
        return self._tts

    async def run_turn(self, session: Session, ws: WebSocket) -> None:
        turn_id = uuid.uuid4().hex[:10]
        metrics = TurnMetrics(turn_id=turn_id)
        session.in_flight = metrics
        session.reset_cancel()
        try:
            await self._do_turn(session, ws, metrics)
        except asyncio.CancelledError:
            log.info("turn.cancelled", extra={"turn_id": turn_id})
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("turn.failed", extra={"turn_id": turn_id})
            try:
                await ws.send_text(msg_error("turn_failed", str(e)))
            except Exception:
                pass
        finally:
            session.in_flight = None

    async def _do_turn(self, session: Session, ws: WebSocket, metrics: TurnMetrics) -> None:
        if session.speech_buffer.size == 0:
            await ws.send_text(msg_error("no_audio", "received stop with no audio"))
            return

        # 1. STT
        await ws.send_text(msg_status("transcribing"))
        metrics.mark("stt_start")
        transcripts = list(self.stt.transcribe(session.speech_buffer, sample_rate=16000, is_final=True))
        text = " ".join(t.text for t in transcripts).strip()
        metrics.mark("stt_done")
        session.reset_speech()
        if not text:
            await ws.send_text(msg_error("no_speech", "could not transcribe audio"))
            return
        await ws.send_text(msg_transcript(text, final=True))
        session.add_user_turn(text)

        if session.cancel_event.is_set():
            return

        # 2. Retrieve
        await ws.send_text(msg_status("retrieving"))
        metrics.mark("rag_start")
        docs = self.retriever.retrieve(text)
        metrics.mark("rag_done")
        if session.cancel_event.is_set():
            return

        # 3. LLM stream + TTS handoff
        await ws.send_text(msg_status("generating"))
        await ws.send_text(msg_tts_meta(self.tts.sample_rate))
        sys_prompt = self.cfg.get("llm", {}).get("system_prompt", DEFAULT_SYSTEM)
        messages = build_messages(text, docs, system=sys_prompt, history=session.history[:-1])
        sb = SentenceBuffer()
        full_answer: list[str] = []
        first_token = False

        async for delta in self.llm.stream(messages):
            if session.cancel_event.is_set():
                return
            if not first_token and delta.text:
                metrics.mark("llm_first_token")
                first_token = True
            if delta.text:
                full_answer.append(delta.text)
                await ws.send_text(msg_answer_delta(delta.text))
                pieces = sb.push(delta.text)
                for p in pieces:
                    await self._speak_piece(p, ws, metrics, session)
            if delta.finished:
                break

        for tail in sb.flush():
            if session.cancel_event.is_set():
                return
            await self._speak_piece(tail, ws, metrics, session)

        metrics.mark("done")
        session.add_assistant_turn("".join(full_answer))
        await ws.send_text(msg_answer_done(metrics.as_dict()))
        log.info("turn.metrics", extra={"turn_id": metrics.turn_id, **metrics.as_dict()})

    async def _speak_piece(self, text: str, ws: WebSocket, metrics: TurnMetrics, session: Session) -> None:
        if not text.strip():
            return
        if "tts_first_byte" not in metrics.marks:
            metrics.mark("tts_first_byte")
        async for pcm in self.tts.synth_stream(text):
            if session.cancel_event.is_set():
                return
            await ws.send_bytes(pcm)
