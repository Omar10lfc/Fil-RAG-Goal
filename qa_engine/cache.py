"""
FilGoalBot — On-disk LLM response cache.

Keyed on (model, intent, ordered chunk_ids, normalised query, prompt_version).
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
import logging
import os
import time
import uuid
from pathlib import Path

from qa_engine import prompts  # PROMPT_VERSION read lazily — see _make_key

log = logging.getLogger("cache")

CACHE_DIR = Path(".cache/llm")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Key version 2: conversation_key folded into the hash (Phase 3). Pre-v2
# entries can never hit again — they miss once and age out via the TTL sweep
# in purge_expired(). Logged at startup by FilGoalRAG.load().
CACHE_KEY_VERSION = 2

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
    conversation_key: str = "",
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "intent": intent,
            # Preserve retrieval order. The LLM context numbers chunks as
            # [1], [2], [3], so the same evidence set in a different order can
            # produce different citation numbers in the answer.
            "chunks": chunk_ids,
            "query": query.strip().lower(),
            # Follow-up context: without this, a cached answer for "من سجل؟"
            # would shadow a later "وماذا عن الشوط الثاني؟" follow-up that
            # resolves the same way. Empty for standalone queries.
            "conv": conversation_key.strip().lower(),
            "key_v": CACHE_KEY_VERSION,
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
        ttl_seconds: int | None = None, conversation_key: str = "") -> str | None:
    """Look up a cached answer. ttl_seconds=None → look up per-intent TTL
    (production default). Pass an explicit value to override (tests use 0
    to force staleness)."""
    if ttl_seconds is None:
        ttl_seconds = ttl_for(intent)
    key = _make_key(model, intent, chunk_ids, query, conversation_key)
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
        answer: str, conversation_key: str = "") -> None:
    key = _make_key(model, intent, chunk_ids, query, conversation_key)
    path = CACHE_DIR / f"{key}.json"
    # Atomic write: dump to a uniquely-named temp file in the same directory,
    # then os.replace() onto the final path. A concurrent get() therefore
    # only ever sees the old entry, no entry, or the complete new entry —
    # never a partially-written JSON file (which would be treated as a miss
    # and the entry silently lost). The uuid suffix prevents two concurrent
    # writers of the same key from clobbering each other's temp file.
    tmp = CACHE_DIR / f"{key}.{uuid.uuid4().hex}.tmp"
    tmp.write_text(
        json.dumps(
            {"ts": time.time(), "answer": answer, "query": query, "intent": intent,
             "prompt_v": prompts.PROMPT_VERSION},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    try:
        os.replace(tmp, path)
    except OSError:
        # Replace failed (e.g. Windows file lock from a concurrent reader).
        # Losing one cache write is harmless; a leftover temp file is not —
        # clean it up.
        try:
            tmp.unlink()
        except OSError:
            pass
    # Size bound (FILGOAL_CACHE_MAX_ENTRIES, default 5,000): evict oldest
    # entries so unique-query traffic can't grow .cache/llm/ forever.
    _enforce_size_cap()


def max_entries() -> int:
    """Size bound for .cache/llm/. Unique-query traffic would otherwise grow
    the directory forever (TTL only *ignores* stale entries; purge_expired()
    only runs at startup). Read per call so tests can monkeypatch the env."""
    try:
        return max(1, int(os.getenv("FILGOAL_CACHE_MAX_ENTRIES", "5000")))
    except ValueError:
        return 5000


def _enforce_size_cap() -> None:
    """Evict oldest-by-mtime entries when the directory exceeds max_entries().
    Best-effort: eviction failures are swallowed — losing the cleanup is
    harmless, crashing the request is not."""
    try:
        files = sorted(CACHE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return
    excess = len(files) - max_entries()
    for stale in files[:max(0, excess)]:
        try:
            stale.unlink()
        except OSError:
            pass


def purge_expired() -> int:
    """Delete cache files whose entry has outlived its intent's TTL, plus any
    that are corrupt/unreadable. Returns the number of files removed.

    The read path (`get`) only ever *ignores* stale entries — it never deletes
    them — so without a sweep the cache directory grows unbounded over a
    long-running deployment. Safe to call at startup: it only removes entries a
    subsequent `get` would already treat as a miss, so it never changes
    behaviour, only reclaims disk."""
    removed = 0
    now = time.time()
    # Orphaned temp files from writes interrupted mid-put() (crash between
    # tmp write and os.replace). Any .tmp older than a minute is dead.
    for tmp in CACHE_DIR.glob("*.tmp"):
        try:
            if now - tmp.stat().st_mtime > 60:
                tmp.unlink()
                removed += 1
        except OSError:
            pass
    for path in CACHE_DIR.glob("*.json"):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            entry = None
        # Unreadable/corrupt → drop it. Otherwise drop only if past its TTL.
        if entry is None or now - entry.get("ts", 0) >= ttl_for(entry.get("intent", "")):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def estimate_tokens(text: str) -> int:
    """Rough token estimate for budget guard. Arabic averages ~3 chars/token
    on llama tokenizers; we use 3 to err conservative (overestimate)."""
    return max(1, len(text) // 3)
