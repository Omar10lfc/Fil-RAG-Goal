"""
FilGoalBot — On-disk LLM response cache.

Keyed on (model, intent, sorted chunk_ids, normalised query). Intentionally
file-based + JSON: trivial to inspect, trivial to invalidate by deletion, and
survives across processes (eval re-runs, API restarts).

Saves Groq tokens on:
  - eval re-runs (same 50 questions, same retrieved chunks)
  - production duplicates (the same query within the eviction window)
"""

import hashlib
import json
import time
from pathlib import Path

CACHE_DIR = Path(".cache/llm")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 30 days. Football news goes stale much faster, but the cache key includes the
# retrieved chunk IDs — when new articles come in, the chunk set shifts and the
# key changes naturally.
DEFAULT_TTL_SECONDS = 30 * 24 * 3600


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
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def get(model: str, intent: str, chunk_ids: list[str], query: str,
        ttl_seconds: int = DEFAULT_TTL_SECONDS) -> str | None:
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
            {"ts": time.time(), "answer": answer, "query": query, "intent": intent},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def estimate_tokens(text: str) -> int:
    """Rough token estimate for budget guard. Arabic averages ~3 chars/token
    on llama tokenizers; we use 3 to err conservative (overestimate)."""
    return max(1, len(text) // 3)
