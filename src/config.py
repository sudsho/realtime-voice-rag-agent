"""config loader. wraps yaml + env overrides into a flat object."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _env_overrides() -> dict:
    """pick up a small set of env vars commonly tweaked at deploy time."""
    out: dict[str, Any] = {}
    if os.getenv("LOG_LEVEL"):
        out.setdefault("app", {})["log_level"] = os.environ["LOG_LEVEL"]
    if os.getenv("LLM_MODEL"):
        out.setdefault("llm", {})["model"] = os.environ["LLM_MODEL"]
    if os.getenv("LLM_PROVIDER"):
        out.setdefault("llm", {})["provider"] = os.environ["LLM_PROVIDER"]
    if os.getenv("STT_MODEL"):
        out.setdefault("stt", {})["model"] = os.environ["STT_MODEL"]
    if os.getenv("TTS_VOICE"):
        out.setdefault("tts", {})["voice"] = os.environ["TTS_VOICE"]
    if os.getenv("CHROMA_DIR"):
        out.setdefault("rag", {})["chroma_dir"] = os.environ["CHROMA_DIR"]
    return out


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else _DEFAULT_PATH
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return _deep_merge(cfg, _env_overrides())


def get(key: str, default: Any = None, cfg: dict | None = None) -> Any:
    """dotted-key lookup, e.g. get('rag.top_k', 6)."""
    cfg = cfg or load_config()
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
