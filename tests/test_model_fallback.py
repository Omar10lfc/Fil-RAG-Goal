"""Unit tests for the big → small model fallback in FilGoalRAG.answer().

The Groq SDK is stubbed so these tests run offline; we monkey-patch
_groq_completion to script the per-call outcomes.
"""

from __future__ import annotations

import shutil
from unittest.mock import MagicMock

import pytest

# Make a real Groq exception class available without importing the SDK.
# rag_pipeline imports the real ones at module load; we just need an
# isolated copy to raise from our patched _groq_completion.
from groq import RateLimitError  # noqa: E402

import qa_engine.rag_pipeline as rag_pipeline
from qa_engine import cache, prompts
from qa_engine.rag_pipeline import (
    ERROR_ANSWER,
    GROQ_MODEL_BIG,
    GROQ_MODEL_SMALL,
    FilGoalRAG,
)


@pytest.fixture(autouse=True)
def isolated_cache_dir(monkeypatch, tmp_path):
    """Keep cache writes from these tests out of the project's .cache/."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    yield
    shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture(autouse=True)
def clean_routing_env(monkeypatch):
    """Routing is env-driven — every case declares its own env so the
    operator's shell (or CI) can't leak FILGOAL_FORCE_* into the suite."""
    monkeypatch.delenv("FILGOAL_FORCE_SMALL_MODEL", raising=False)
    monkeypatch.delenv("FILGOAL_FORCE_BIG_MODEL", raising=False)


def _build_rag_with_fake_retriever(chunks: list[dict]) -> FilGoalRAG:
    """Construct a FilGoalRAG instance whose retriever returns `chunks` and
    whose Groq client is a no-op stub. The Groq call itself is patched
    per-test via _groq_completion."""
    retriever = MagicMock()
    retriever.retrieve.return_value = chunks
    rag = FilGoalRAG(retriever=retriever)
    rag.groq = MagicMock()  # avoid the "load() not called" assertion
    return rag


def _make_rate_limit_error() -> RateLimitError:
    """Construct a RateLimitError without going over the wire. The SDK's
    initialiser takes (message, response, body); we don't care about
    payload contents — only that `isinstance(e, RateLimitError)` holds."""
    return RateLimitError(
        "rate limit",
        response=MagicMock(status_code=429),
        body=None,
    )


def _chunk(i: int = 0) -> dict:
    return {
        "chunk_id":     f"c{i}",
        "title":        "T",
        "pub_date":     "2026-04-01",
        "body_clean":   "x",
        "article_type": "team_news",
        "league":       "",
        "source_url":   "",
        "_rrf_score":   0.5,
    }


def test_fallback_to_small_model_on_big_model_rate_limit():
    """Intent routes to big model → 429 → pipeline retries on small model.
    Verify the result reports the small model as the effective model and sets the
    model_fallback flag."""
    rag = _build_rag_with_fake_retriever([_chunk()])
    call_log: list[str] = []

    def fake_completion(model: str, system_prompt: str, user_prompt: str) -> str:
        call_log.append(model)
        if model == GROQ_MODEL_BIG:
            raise _make_rate_limit_error()
        return "small-model answer"

    rag._groq_completion = fake_completion  # type: ignore[method-assign]

    # team_news is non-extractive -> large model by default.
    result = rag.answer("ما أخبار مران الأهلي؟")

    assert call_log == [GROQ_MODEL_BIG, GROQ_MODEL_SMALL], \
        "should have tried the big model first, then the small model as fallback"
    assert result["answer"]         == "small-model answer"
    assert result["model"]          == GROQ_MODEL_SMALL
    assert result["model_fallback"] is True
    assert result["cache_reason"]   == "miss"


def test_no_fallback_when_already_on_small_model():
    """Extractive intents route to the small model directly. A 429 there has nowhere to
    fall back to — surface the error, don't pretend to succeed."""
    rag = _build_rag_with_fake_retriever([_chunk()])
    call_log: list[str] = []

    def fake_completion(model: str, system_prompt: str, user_prompt: str) -> str:
        call_log.append(model)
        raise _make_rate_limit_error()

    rag._groq_completion = fake_completion  # type: ignore[method-assign]

    # match_result is extractive → routed to the small model from the start.
    result = rag.answer("ما نتيجة مباراة الأهلي؟")

    assert call_log == [GROQ_MODEL_SMALL], "must not loop back to the small model on itself"
    assert result["answer"]         == ERROR_ANSWER
    assert result["model_fallback"] is False
    assert result["cache_reason"]   == "skipped_rate_limit"


def test_both_models_rate_limited():
    """Big model 429 → fallback to small model → small model also 429. The
    pipeline must not cache the error string."""
    rag = _build_rag_with_fake_retriever([_chunk()])
    call_log: list[str] = []

    def fake_completion(model: str, system_prompt: str, user_prompt: str) -> str:
        call_log.append(model)
        raise _make_rate_limit_error()

    rag._groq_completion = fake_completion  # type: ignore[method-assign]

    result = rag.answer("ما أخبار مران الأهلي؟")

    assert call_log == [GROQ_MODEL_BIG, GROQ_MODEL_SMALL]
    assert result["answer"]         == ERROR_ANSWER
    assert result["cache_reason"]   == "skipped_rate_limit"
    # Crucially: the error answer must NOT be cached, or this query would
    # poison its cache key for the entire intent TTL.
    chunk_ids = ["c0"]
    assert cache.get(GROQ_MODEL_BIG,   "team_news", chunk_ids, "ما أخبار مران الأهلي؟") is None
    assert cache.get(GROQ_MODEL_SMALL, "team_news", chunk_ids, "ما أخبار مران الأهلي؟") is None


def test_fallback_answer_is_cached_under_small_model_key():
    """When fallback succeeds, cache the answer under the model that
    actually produced it (small) — not under the intended model (big)."""
    rag = _build_rag_with_fake_retriever([_chunk(1)])
    _ = prompts.PROMPT_VERSION  # touch so the import isn't pruned

    def fake_completion(model: str, system_prompt: str, user_prompt: str) -> str:
        if model == GROQ_MODEL_BIG:
            raise _make_rate_limit_error()
        return "fallback answer"

    rag._groq_completion = fake_completion  # type: ignore[method-assign]
    rag.answer("ما أخبار مران الأهلي؟")

    chunk_ids = ["c1"]
    assert cache.get(GROQ_MODEL_SMALL, "team_news", chunk_ids, "ما أخبار مران الأهلي؟") == "fallback answer"
    assert cache.get(GROQ_MODEL_BIG,   "team_news", chunk_ids, "ما أخبار مران الأهلي؟") is None


def test_force_big_model_env_routes_extractive_to_big(monkeypatch):
    """FILGOAL_FORCE_BIG_MODEL=1 routes even extractive intents to the big
    model (the operator-visible replacement for the old pytest-sniffing hack)."""
    monkeypatch.setenv("FILGOAL_FORCE_BIG_MODEL", "1")
    rag = _build_rag_with_fake_retriever([_chunk()])
    call_log: list[str] = []

    def fake_completion(model: str, system_prompt: str, user_prompt: str) -> str:
        call_log.append(model)
        return "big answer"

    rag._groq_completion = fake_completion  # type: ignore[method-assign]

    # match_result is extractive → would use the small model without the flag.
    result = rag.answer("ما نتيجة مباراة الأهلي؟")

    assert call_log == [GROQ_MODEL_BIG]
    assert result["model"] == GROQ_MODEL_BIG
    assert result["model_fallback"] is False


def test_empty_completion_retries_once_on_big_model():
    """Empty completion on the small model (the gpt-oss-20b runaway-reasoning
    signature) → exactly one retry on the big model."""
    rag = _build_rag_with_fake_retriever([_chunk()])
    call_log: list[str] = []

    def fake_completion(model: str, system_prompt: str, user_prompt: str) -> str:
        call_log.append(model)
        if model == GROQ_MODEL_SMALL:
            return "   "  # empty completion, as the buggy 20b returns
        return "big-model answer"

    rag._groq_completion = fake_completion  # type: ignore[method-assign]

    result = rag.answer("ما نتيجة مباراة الأهلي؟")

    assert call_log == [GROQ_MODEL_SMALL, GROQ_MODEL_BIG]
    assert result["answer"]         == "big-model answer"
    assert result["model"]          == GROQ_MODEL_BIG
    assert result["model_fallback"] is True


def test_force_big_rate_limit_returns_error_without_small_fallback(monkeypatch):
    """Under FILGOAL_FORCE_BIG_MODEL=1 a 429 on big surfaces the error — the
    operator ruled the small model out, so falling back to it is wrong."""
    monkeypatch.setenv("FILGOAL_FORCE_BIG_MODEL", "1")
    rag = _build_rag_with_fake_retriever([_chunk()])
    call_log: list[str] = []

    def fake_completion(model: str, system_prompt: str, user_prompt: str) -> str:
        call_log.append(model)
        raise _make_rate_limit_error()

    rag._groq_completion = fake_completion  # type: ignore[method-assign]

    result = rag.answer("ما أخبار مران الأهلي؟")

    assert call_log == [GROQ_MODEL_BIG], "must not touch the ruled-out small model"
    assert result["answer"]         == ERROR_ANSWER
    assert result["model_fallback"] is False
    assert result["cache_reason"]   == "skipped_rate_limit"


def test_no_retry_on_second_failure():
    """Small model empty → big retry also fails → ERROR_ANSWER after exactly
    two calls. Failures never loop."""
    rag = _build_rag_with_fake_retriever([_chunk()])
    call_log: list[str] = []

    def fake_completion(model: str, system_prompt: str, user_prompt: str) -> str:
        call_log.append(model)
        if model == GROQ_MODEL_SMALL:
            raise RuntimeError("empty completion content")
        raise RuntimeError("empty completion content")

    rag._groq_completion = fake_completion  # type: ignore[method-assign]

    result = rag.answer("ما نتيجة مباراة الأهلي؟")

    assert call_log == [GROQ_MODEL_SMALL, GROQ_MODEL_BIG]
    assert result["answer"]         == ERROR_ANSWER
    assert result["model_fallback"] is False
    assert result["cache_reason"]   == "skipped_error"
    chunk_ids = ["c0"]
    assert cache.get(GROQ_MODEL_BIG,   "match_result", chunk_ids, "ما نتيجة مباراة الأهلي؟") is None
    assert cache.get(GROQ_MODEL_SMALL, "match_result", chunk_ids, "ما نتيجة مباراة الأهلي؟") is None


def test_match_result_prepare_omits_none_date_filters(monkeypatch):
    """When only one bound is extracted, _prepare should not pass None-valued
    date filters into retriever kwargs."""
    rag = _build_rag_with_fake_retriever([])
    monkeypatch.setattr(rag_pipeline, "detect_intent", lambda _query: "match_result")
    monkeypatch.setattr(rag_pipeline, "extract_date_window", lambda _query: (None, "2026-05-10"))

    _ = rag._prepare("q")

    kwargs = rag.retriever.retrieve.call_args.kwargs
    assert kwargs["filter_type"] == "match_result"
    assert kwargs["date_to"] == "2026-05-10"
    assert "date_from" not in kwargs
