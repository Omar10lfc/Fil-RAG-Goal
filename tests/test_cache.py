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


def test_chunk_id_order_affects_key():
    cache.put("m", "i", ["a", "b", "c"], "q", "ans")
    assert cache.get("m", "i", ["c", "b", "a"], "q") is None


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


def test_per_intent_ttl_match_result_shorter_than_player_info():
    """match_result facts go stale within hours; player bios within weeks.
    The TTL map encodes that asymmetry."""
    assert cache.ttl_for("match_result") < cache.ttl_for("player_info")
    assert cache.ttl_for("lineup")       < cache.ttl_for("transfer_news")


def test_unknown_intent_falls_back_to_default_ttl():
    assert cache.ttl_for("not_a_real_intent") == cache.DEFAULT_TTL_SECONDS


def test_purge_expired_removes_stale_keeps_fresh():
    """purge_expired() deletes entries past their intent TTL but leaves fresh
    ones — so the read behaviour is unchanged, only disk is reclaimed."""
    import time

    # Fresh entry under a long-TTL intent — must survive the purge.
    cache.put("m", "player_info", ["a"], "q-fresh", "fresh-answer")
    # Stale entry under a short-TTL intent — backdate its timestamp well past
    # the match_result TTL so the sweep removes it.
    cache.put("m", "match_result", ["b"], "q-stale", "stale-answer")
    stale_key = cache._make_key("m", "match_result", ["b"], "q-stale")
    stale_path = cache.CACHE_DIR / f"{stale_key}.json"
    import json
    entry = json.loads(stale_path.read_text(encoding="utf-8"))
    entry["ts"] = time.time() - (cache.ttl_for("match_result") + 60)
    stale_path.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")

    removed = cache.purge_expired()

    assert removed == 1
    assert cache.get("m", "player_info", ["a"], "q-fresh") == "fresh-answer"
    assert not stale_path.exists()


def test_purge_expired_removes_corrupt_files():
    """A corrupt cache file (unreadable JSON) is dropped by the sweep."""
    bad = cache.CACHE_DIR / "deadbeef.json"
    bad.write_text("{not valid json", encoding="utf-8")
    assert cache.purge_expired() == 1
    assert not bad.exists()


def test_prompt_version_invalidates_cache(monkeypatch):
    """Bumping PROMPT_VERSION must shift the cache key so old entries
    are no longer found — otherwise prompt edits would be silently
    shadowed by stale completions."""
    from qa_engine import prompts
    monkeypatch.setattr(prompts, "PROMPT_VERSION", 1)
    cache.put("m", "i", ["a"], "q", "old-answer")
    assert cache.get("m", "i", ["a"], "q") == "old-answer"
    monkeypatch.setattr(prompts, "PROMPT_VERSION", 2)
    # Same inputs, new prompt version → key changes → miss.
    assert cache.get("m", "i", ["a"], "q") is None


def test_conversation_key_shifts_cache():
    """A cached standalone answer must not shadow a follow-up: the follow-up
    context participates in the key, so identical (model, intent, chunks,
    query) with different conversation context are different entries."""
    cache.put("m", "i", ["a"], "q", "standalone-answer")
    assert cache.get("m", "i", ["a"], "q") == "standalone-answer"
    # Same query + a follow-up context → miss (would have shadowed before A13).
    assert cache.get("m", "i", ["a"], "q", conversation_key="prev exchange") is None
    cache.put("m", "i", ["a"], "q", "followup-answer", conversation_key="prev exchange")
    assert cache.get("m", "i", ["a"], "q", conversation_key="prev exchange") == "followup-answer"
    # And the standalone entry is untouched.
    assert cache.get("m", "i", ["a"], "q") == "standalone-answer"


def test_size_cap_evicts_oldest_entries(monkeypatch):
    """FILGOAL_CACHE_MAX_ENTRIES bounds .cache/llm/: beyond the cap, the
    oldest-by-mtime entries are evicted and the newest survives."""
    import time

    monkeypatch.setenv("FILGOAL_CACHE_MAX_ENTRIES", "3")
    for i in range(5):
        cache.put("m", "i", [f"c{i}"], f"q{i}", f"ans{i}")
        # Distinct mtimes so "oldest" is deterministic on coarse filesystems.
        time.sleep(0.02)

    files = list(cache.CACHE_DIR.glob("*.json"))
    assert len(files) <= 3
    # Newest entry survives; oldest is gone.
    assert cache.get("m", "i", ["c4"], "q4") == "ans4"
    assert cache.get("m", "i", ["c0"], "q0") is None
