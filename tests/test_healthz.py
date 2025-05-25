"""tests for healthz/readyz endpoints."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.healthz import router


def _client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_healthz_returns_ok():
    c = _client()
    r = c.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # config keys should be present (even if None in some envs)
    assert "stt_model" in body
    assert "llm_provider" in body


def test_readyz_returns_ready():
    c = _client()
    r = c.get("/readyz")
    assert r.status_code == 200
    assert r.json() == {"ready": True}
