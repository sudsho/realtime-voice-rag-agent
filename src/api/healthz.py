"""richer health endpoint for ALB / smoke tests."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from src.config import load_config


router = APIRouter()


@router.get("/healthz")
async def healthz() -> Dict[str, Any]:
    cfg = load_config()
    return {
        "status": "ok",
        "stt_model": cfg.get("stt", {}).get("model"),
        "llm_provider": cfg.get("llm", {}).get("provider"),
        "tts_provider": cfg.get("tts", {}).get("provider"),
    }


@router.get("/readyz")
async def readyz() -> Dict[str, Any]:
    # cheap readiness; the heavy model warming is done at startup if WARM_MODELS=1
    return {"ready": True}
