from src.agent.prompt import build_messages, render_context
from src.rag.store import RetrievedDoc


def _docs():
    return [
        RetrievedDoc(text="refunds within 14 days", source="01-billing.md", chunk_id="b1", score=0.9, metadata={}),
        RetrievedDoc(text="2FA via TOTP", source="02-account.md", chunk_id="a1", score=0.8, metadata={}),
    ]


def test_render_context_handles_empty():
    assert "no relevant" in render_context([]).lower()


def test_render_context_includes_source_tag():
    out = render_context(_docs())
    assert "01-billing.md" in out
    assert "[1]" in out and "[2]" in out


def test_build_messages_has_system_user_only_without_history():
    msgs = build_messages("how do refunds work?", _docs(), system="SYS")
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "SYS"
    assert msgs[-1]["role"] == "user"
    assert "refunds work" in msgs[-1]["content"]


def test_build_messages_threads_history():
    history = [
        {"role": "user", "content": "earlier turn"},
        {"role": "assistant", "content": "earlier reply"},
    ]
    msgs = build_messages("now", _docs(), history=history)
    assert msgs[1] == {"role": "user", "content": "earlier turn"}
    assert msgs[2] == {"role": "assistant", "content": "earlier reply"}
