"""Unit tests for the context builder in rag_pipeline. Pure function — no
model loads needed."""

import importlib.util
from pathlib import Path

# Importing qa_engine.rag_pipeline triggers Groq init. Pull just _build_context
# directly without going through the package.
spec = importlib.util.spec_from_file_location(
    "rag_pipeline_module",
    Path(__file__).resolve().parents[1] / "qa_engine" / "rag_pipeline.py",
)


def _import_build_context():
    # Lazy module load with mocked-out imports the function doesn't need.
    import sys, types
    if "qa_engine.rag_pipeline" in sys.modules:
        return sys.modules["qa_engine.rag_pipeline"]._build_context

    # Stub Groq + retriever before import so we don't pay their init cost.
    sys.modules.setdefault("groq", types.ModuleType("groq"))
    sys.modules["groq"].Groq = lambda **kw: None
    from qa_engine.rag_pipeline import _build_context
    return _build_context


def test_build_context_uses_body_clean():
    _build_context = _import_build_context()
    chunks = [{
        "title":       "خبر تجريبي",
        "pub_date":    "2026-04-22T12:00:00",
        "body_clean":  "هذا نص خبر مقروء بالعربية الصحيحة",
        "text":        "هذا نص مطبع لا يجب أن يظهر",
        "source_url":  "https://example.com/1",
        "article_type": "match_result",
        "league":      "egyptian_league",
    }]
    context, sources = _build_context(chunks)
    assert "هذا نص خبر مقروء" in context
    assert "نص مطبع" not in context, "body_clean must be preferred over normalised text"
    assert sources[0]["url"] == "https://example.com/1"
    assert sources[0]["pub_date"] == "2026-04-22"


def test_build_context_lineup_gets_more_room():
    _build_context = _import_build_context()
    long_body = "ا" * 2000
    lineup_chunk = {"title": "T", "pub_date": "", "body_clean": long_body,
                    "article_type": "lineup", "league": "", "source_url": ""}
    news_chunk   = {"title": "T", "pub_date": "", "body_clean": long_body,
                    "article_type": "team_news", "league": "", "source_url": ""}
    lineup_ctx, _ = _build_context([lineup_chunk])
    news_ctx, _   = _build_context([news_chunk])
    # lineups get 1500 chars, news gets 800 → lineup context is longer
    assert len(lineup_ctx) > len(news_ctx)


def test_build_context_numbers_chunks_for_citations():
    _build_context = _import_build_context()
    chunks = [
        {"title": "أ", "pub_date": "", "body_clean": "x", "article_type": "", "league": "", "source_url": ""},
        {"title": "ب", "pub_date": "", "body_clean": "y", "article_type": "", "league": "", "source_url": ""},
    ]
    context, _ = _build_context(chunks)
    assert "[1]" in context and "[2]" in context
