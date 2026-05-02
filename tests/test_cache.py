"""Unit tests for the LLM disk cache."""

import shutil
from pathlib import Path

import pytest

from qa_engine import cache


@pytest.fixture(autouse=True)
def isolated_cache_dir(monkeypatch, tmp_path):
    """Redirect the cache to a per-test temp dir so tests don't pollute .cache/."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    yield
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_get_returns_none_on_miss():
    assert cache.get("model", "intent", ["c1"], "q") is None


def test_put_then_get_round_trips():
    cache.put("model", "intent", ["c1", "c2"], "ما هي النتيجة؟", "1-1")
    assert cache.get("model", "intent", ["c1", "c2"], "ما هي النتيجة؟") == "1-1"


def test_chunk_id_order_does_not_affect_key():
    cache.put("m", "i", ["a", "b", "c"], "q", "ans")
    assert cache.get("m", "i", ["c", "b", "a"], "q") == "ans"


def test_query_normalisation_in_key():
    cache.put("m", "i", ["a"], "  Hello  ", "ans")
    assert cache.get("m", "i", ["a"], "hello") == "ans"


def test_different_models_get_different_keys():
    cache.put("70b", "i", ["a"], "q", "big-answer")
    cache.put("8b",  "i", ["a"], "q", "small-answer")
    assert cache.get("70b", "i", ["a"], "q") == "big-answer"
    assert cache.get("8b",  "i", ["a"], "q") == "small-answer"


def test_ttl_zero_is_always_stale():
    cache.put("m", "i", ["a"], "q", "ans")
    assert cache.get("m", "i", ["a"], "q", ttl_seconds=0) is None


def test_estimate_tokens_is_positive():
    assert cache.estimate_tokens("") == 1
    assert cache.estimate_tokens("hello world") >= 1
    assert cache.estimate_tokens("a" * 300) > cache.estimate_tokens("a" * 30)
