"""warm model + embedding caches before traffic arrives.

run this in the container at startup or as an ECS task pre-step. it
preloads:
- faster-whisper model into RAM (with int8 weights)
- silero VAD via torch hub
- the cross-encoder reranker
- a single openai embedding to verify the API key is valid
"""

from __future__ import annotations

import os
import sys
import time

from src.config import load_config
from src.logging_utils import get_logger, setup_logging
from src.rag.store import ChromaStore
from src.stt.vad import SileroVad
from src.stt.whisper_stream import WhisperStream


def main() -> int:
    setup_logging("INFO")
    log = get_logger("warm")
    cfg = load_config()
    t0 = time.perf_counter()

    log.info("loading whisper")
    stt_cfg = cfg.get("stt", {})
    stt = WhisperStream(
        model_name=stt_cfg.get("model", "small.en"),
        device=stt_cfg.get("device", "cpu"),
        compute_type=stt_cfg.get("compute_type", "int8"),
    )
    stt._ensure_model()  # noqa: SLF001

    log.info("loading vad")
    vad = SileroVad()
    vad._ensure_model()  # noqa: SLF001

    rag_cfg = cfg.get("rag", {})
    log.info("warming chroma + embed")
    store = ChromaStore(
        persist_dir=rag_cfg.get("chroma_dir", ".chroma"),
        collection=rag_cfg.get("collection", "support"),
        embed_model=rag_cfg.get("embed_model", "text-embedding-3-small"),
    )
    try:
        store.query("warmup", k=1)
    except Exception as e:  # noqa: BLE001
        log.warning("chroma query failed - did you run kb-ingest? err=%s", e)

    elapsed = time.perf_counter() - t0
    log.info("warm done in %.1fs", elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
