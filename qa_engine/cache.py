"""
FilGoalBot — On-disk LLM response cache.

Keyed on (model, intent, sorted chunk_ids, normalised query, prompt_version).
Intentionally file-based + JSON: trivial to inspect, trivial to invalidate by
deletion, and survives across processes (eval re-runs, API restarts).

TTL is intent-aware. Match results go stale in hours; player bios stay fresh
for weeks. See INTENT_TTL_SECONDS.

Saves Groq tokens on:
  - eval re-runs (same 50 questions, same retrieved chunks)
  - production duplicates (the same query within the eviction window)
"""

import hashlib
import json
import time
from pathlib import Path

from qa_engine import prompts  # PROMPT_VERSION read lazily — see _make_key

CACHE_DIR = Path(".cache/llm")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Default for unknown intents and ablation tests. Real intents are looked up
# in INTENT_TTL_SECONDS below.
DEFAULT_TTL_SECONDS = 30 * 24 * 3600

# Intent-aware TTLs. The cache key includes chunk_ids — so when fresh
# articles arrive the key shifts and stale entries are skipped naturally
# — but TTL is the safety net for cases where the corpus is static
# (e.g. eval re-runs) and the underlying facts move on.
#
# Tuned to the rate at which each intent's facts go stale in reality:
#   match_result : a score changes once, but lineup/scorer chatter
#                  evolves through the day → short window.
#   lineup       : valid until kickoff, sometimes 12–48h.
#   team_news    : presser quotes age out in a few days.
#   transfer_news: rumours linger for a window, finalised deals are
#                  stable once announced → a week is the right middle.
#   player_info  : bios and stats are slow-moving.
#   general      : true trivia (rules, history) effectively immortal.
INTENT_TTL_SECONDS: dict[str, int] = {
    "match_result":     6 * 3600,
    "lineup":          12 * 3600,
    "team_news":        3 * 24 * 3600,
    "transfer_news":    7 * 24 * 3600,
    "player_info":     14 * 24 * 3600,
    "general_football": 30 * 24 * 3600,
}


def ttl_for(intent: str) -> int:
    """Resolve the cache TTL for an intent, falling back to the default."""
    return INTENT_TTL_SECONDS.get(intent, DEFAULT_TTL_SECONDS)


def _make_key(
    model: str,
    intent: str,
    chunk_ids: list[str],
    query: str,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "intent": intent,
            "chunks": sorted(chunk_ids),
            "query": query.strip().lower(),
            # PROMPT_VERSION folded in so editing prompts.py auto-invalidates
            # every prior cached answer. Without this, a prompt rewrite would
            # be shadowed by stale completions until the TTL expired.
            # Looked up at call time, not import time, so a runtime bump
            # of prompts.PROMPT_VERSION (and tests that monkeypatch it)
            # actually shifts the key.
            "prompt_v": prompts.PROMPT_VERSION,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def get(model: str, intent: str, chunk_ids: list[str], query: str,
        ttl_seconds: int | None = None) -> str | None:
    """Look up a cached answer. ttl_seconds=None → look up per-intent TTL
    (production default). Pass an explicit value to override (tests use 0
    to force staleness)."""
    if ttl_seconds is None:
        ttl_seconds = ttl_for(intent)
    key = _make_key(model, intent, chunk_ids, query)
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if time.time() - entry.get("ts", 0) >= ttl_seconds:
        return None
    return entry.get("answer")


def put(model: str, intent: str, chunk_ids: list[str], query: str,
        answer: str) -> None:
    key = _make_key(model, intent, chunk_ids, query)
    path = CACHE_DIR / f"{key}.json"
    path.write_text(
        json.dumps(
            {"ts": time.time(), "answer": answer, "query": query, "intent": intent,
             "prompt_v": prompts.PROMPT_VERSION},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def estimate_tokens(text: str) -> int:
    """Rough token estimate for budget guard. Arabic averages ~3 chars/token
    on llama tokenizers; we use 3 to err conservative (overestimate)."""
    return max(1, len(text) // 3)
