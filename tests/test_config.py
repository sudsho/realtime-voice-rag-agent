"""tests for config loader (yaml + env override + dotted lookup)."""

from __future__ import annotations

import textwrap

import pytest

from src import config as cfg_mod


@pytest.fixture(autouse=True)
def _clear_cache():
    cfg_mod.load_config.cache_clear()
    yield
    cfg_mod.load_config.cache_clear()


def _write(tmp_path, text: str):
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return str(p)


def test_load_basic(tmp_path):
    p = _write(tmp_path, """
        app:
          name: test
        rag:
          top_k: 4
    """)
    c = cfg_mod.load_config(p)
    assert c["app"]["name"] == "test"
    assert c["rag"]["top_k"] == 4


def test_env_overrides_yaml(tmp_path, monkeypatch):
    p = _write(tmp_path, """
        llm:
          provider: openai
          model: gpt-4o-mini
    """)
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    c = cfg_mod.load_config(p)
    assert c["llm"]["model"] == "gpt-4o"
    assert c["llm"]["provider"] == "openai"  # untouched


def test_dotted_get(tmp_path):
    p = _write(tmp_path, """
        rag:
          top_k: 7
    """)
    c = cfg_mod.load_config(p)
    assert cfg_mod.get("rag.top_k", cfg=c) == 7
    assert cfg_mod.get("rag.missing", default=99, cfg=c) == 99
    assert cfg_mod.get("nope.deep.key", default=None, cfg=c) is None


def test_deep_merge_nested(tmp_path, monkeypatch):
    p = _write(tmp_path, """
        rag:
          top_k: 6
          chroma_dir: .chroma
    """)
    monkeypatch.setenv("CHROMA_DIR", "/data/.chroma")
    c = cfg_mod.load_config(p)
    assert c["rag"]["chroma_dir"] == "/data/.chroma"
    assert c["rag"]["top_k"] == 6
