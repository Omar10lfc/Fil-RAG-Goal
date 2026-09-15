"""API endpoint tests: /ask passthrough, /ask/stream SSE, 429, /health.

The Groq/FAISS stack is stubbed — api.main.rag is replaced with a MagicMock
so no model loads and no network is touched.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

os.environ.setdefault("GROQ_API_KEY", "gsk_test_dummy_key_for_ci_runs_only_xxxxxxxx")

import pytest
from fastapi.testclient import TestClient

import api.main as api_main


@pytest.fixture()
def mock_rag(monkeypatch):
    rag = MagicMock()
    rag.groq = MagicMock()
    rag.retriever = MagicMock()
    rag.retriever.metadata = [{}, {}]
    rag.retriever.retrieve.return_value = [{"title": "t"}]
    monkeypatch.setattr(api_main, "rag", rag)
    return rag


@pytest.fixture()
def client():
    # NOTE: not used as a context manager on purpose — entering the context
    # would run the lifespan and overwrite api_main.rag with a real FilGoalRAG.
    return TestClient(api_main.app)


def _answer(idx: int = 0) -> dict:
    return {
        "answer": f"answer-{idx}",
        "intent": "team_news",
        "sources": [{
            "title": "T", "url": "https://example.com", "pub_date": "2026-04-01",
            "article_type": "article", "league": "", "chunk_id": "c0", "rrf_score": 0.5,
        }],
        "model": "big", "model_fallback": False, "cached": False,
        "cache_reason": "miss", "retrieval_ms": 10, "llm_ms": 20, "n_chunks": 1,
    }


def test_ask_passes_conversation_context_to_pipeline(mock_rag, client):
    """conversation_context must reach rag.answer() (follow-ups were
    unreachable before — pipeline accepted the arg, the API never sent it)."""
    mock_rag.answer.return_value = _answer()

    r = client.post("/ask",
                    json={"query": "وماذا حدث بعدها؟",
                          "conversation_context": "السؤال السابق: …"},
                    headers={"X-Forwarded-For": "10.1.0.1"})

    assert r.status_code == 200
    _, kwargs = mock_rag.answer.call_args
    assert kwargs.get("conversation_context") == "السؤال السابق: …"
    body = r.json()
    assert body["answer"] == "answer-0"
    assert body["intent"] == "team_news"
    assert body["request_id"]


def test_ask_stream_event_sequence(mock_rag, client):
    """SSE order: meta (intent/model/sources) → delta(s) → done (latency,
    cache_reason, model_fallback)."""
    partial = {**_answer(0), "answer": "answer", "llm_ms": 0}
    final = _answer(0)
    mock_rag.answer_stream.return_value = iter([partial, final])

    r = client.post("/ask/stream",
                    json={"query": "ما الخبر؟"},
                    headers={"X-Forwarded-For": "10.2.0.1"})

    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]

    events: list[tuple[str, str]] = []
    for block in r.text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        kind, _, data = block.partition("\ndata: ")
        events.append((kind.replace("event: ", ""), data))

    kinds = [k for k, _ in events]
    assert kinds[0] == "meta"
    assert "delta" in kinds
    assert kinds[-1] == "done"

    import json as _json
    meta = _json.loads(events[0][1])
    assert meta["intent"] == "team_news"
    assert meta["sources"][0]["chunk_id"] == "c0"
    done = _json.loads(events[-1][1])
    assert done["cache_reason"] == "miss"
    assert done["model_fallback"] is False
    assert done["request_id"]


def test_ask_rate_limited_after_window_exhausted(mock_rag, client):
    """20/minute default: the 21st request from one client IP gets a 429."""
    mock_rag.answer.return_value = _answer()
    headers = {"X-Forwarded-For": "10.3.0.99"}  # fresh bucket for this test
    statuses = set()
    for _ in range(30):
        r = client.post("/ask", json={"query": "خبر؟"}, headers=headers)
        statuses.add(r.status_code)
        if r.status_code == 429:
            break
    assert 429 in statuses


def test_health_ok(mock_rag, client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
