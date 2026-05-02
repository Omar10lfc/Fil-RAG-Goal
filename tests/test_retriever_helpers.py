"""Unit tests for the retriever's pure helpers — tokenizer, normaliser,
recency boost. None of these need FAISS or sentence-transformers."""

from datetime import datetime, timedelta, timezone

import pytest

from retrieval.hybrid_retriever import (
    _normalize_ar,
    _strip_clitics,
    _tokenize,
    _recency_multiplier,
    RECENCY_MAX_BOOST,
)


def test_normalize_strips_tashkeel():
    assert _normalize_ar("مُحَمَّدٌ") == "محمد"


def test_normalize_unifies_alef_variants():
    for variant in ["أحمد", "إحمد", "آحمد", "ٱحمد"]:
        assert _normalize_ar(variant) == "احمد"


def test_normalize_collapses_ya_and_ta_marbouta():
    assert _normalize_ar("الرياضة") == "الرياضه"
    assert _normalize_ar("الكرى")    == "الكري"


def test_strip_clitics_removes_definite_article():
    assert _strip_clitics("الأهلي") == "اهلي" or _strip_clitics("الاهلي") == "اهلي"


def test_strip_clitics_keeps_short_tokens_intact():
    # "في" is 2 chars; stripping "ف" would leave "ي" which is too short — keep as-is
    assert _strip_clitics("في") == "في"


def test_tokenize_drops_one_char_tokens():
    tokens = _tokenize("ا ب الأهلي")
    assert "ا" not in tokens and "ب" not in tokens
    assert any("اهلي" in t or "الاهلي" in t for t in tokens)


def test_recency_multiplier_recent_is_max():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    mult = _recency_multiplier(today)
    assert mult == pytest.approx(1.0 + RECENCY_MAX_BOOST, rel=0.01)


def test_recency_multiplier_old_decays_to_one():
    very_old = (datetime.now(timezone.utc) - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    mult = _recency_multiplier(very_old)
    assert mult == pytest.approx(1.0, abs=0.01)


def test_recency_multiplier_handles_missing_or_malformed():
    assert _recency_multiplier("") == 1.0
    assert _recency_multiplier("not-a-date") == 1.0
