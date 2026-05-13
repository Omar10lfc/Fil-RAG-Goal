"""
Structured-logging configuration for FilGoalBot.

Two modes, selected by the FILGOAL_LOG_FORMAT env var:
  - "text" (default) — human-readable lines, kept for local dev.
  - "json"           — one JSON object per line, for shipping to log
                       aggregators (Cloud Logging, Datadog, Loki …).

The JSON formatter pulls a fixed set of structured fields off the log
record's `extra=` mapping (request_id, intent, model, cache_reason,
retrieval_ms, llm_ms, latency_ms, n_chunks) so dashboards can chart
latency-vs-cache-hit-rate without parsing free-text messages.
"""

from __future__ import annotations

import json
import logging
import os

# Fields callers may attach via `log.info(msg, extra={...})`. Anything not
# listed here is dropped from the JSON record to keep payloads predictable.
_STRUCTURED_FIELDS: tuple[str, ...] = (
    "request_id",
    "intent",
    "model",
    "cached",
    "cache_reason",
    "retrieval_ms",
    "llm_ms",
    "latency_ms",
    "n_chunks",
    "query_len",
)


class JSONFormatter(logging.Formatter):
    """One-line JSON-per-record formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts":     self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }
        for field in _STRUCTURED_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            # Keep a single-line representation — don't dump the full
            # stack (see api/main.py: we deliberately avoid exc_info on
            # paths that may hold the Groq client).
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    """Install the chosen formatter on the root logger. Idempotent — safe
    to call from both module-import (api/main.py) and __main__ blocks."""
    fmt = os.getenv("FILGOAL_LOG_FORMAT", "text").lower()
    root = logging.getLogger()
    root.setLevel(level)

    # Wipe existing handlers so a re-import (e.g. by uvicorn --reload)
    # doesn't double-log every line.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler()
    if fmt == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
    root.addHandler(handler)
