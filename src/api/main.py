"""FastAPI app + WebSocket endpoint."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.agent.pipeline import VoicePipeline
from src.api.healthz import router as healthz_router
from src.config import load_config
from src.logging_utils import get_logger, setup_logging
from src.ws.server import VoiceWsHandler


log = get_logger(__name__)
_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg = load_config()
    setup_logging(cfg.get("app", {}).get("log_level", "INFO"))
    pipeline = VoicePipeline(cfg)
    if os.getenv("WARM_MODELS", "0") == "1":
        log.info("warming models")
        pipeline.warm()
    app.state.pipeline = pipeline
    app.state.ws_handler = VoiceWsHandler(pipeline)
    log.info("api.startup", extra={"version": "0.1.0"})
    yield
    log.info("api.shutdown")


app = FastAPI(title="voice-rag-agent", lifespan=lifespan)
app.include_router(healthz_router)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(_FRONTEND / "index.html"))


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket) -> None:
    await app.state.ws_handler(ws)


# static frontend assets (JS, CSS) - mounted only if the dir exists
if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")
