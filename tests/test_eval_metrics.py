"""Unit tests for the eval metrics. Embedding similarity test is opt-in
because it loads a 400MB model — gate it on the env var EVAL_FULL=1."""

import os

import pytest

from evaluation.evaluate import (
    rouge1_f1,
    keyword_hit,
    mrr_score,
    intent_accuracy,
    _normalize_for_metric,
)


def test_normalize_unifies_arabic_surface_forms():
    # The whole point: "النتيجة" and "النتيجه" must compare equal after normalisation
    assert _normalize_for_metric("النتيجة") == _normalize_for_metric("النتيجه")
    assert _normalize_for_metric("أحمد") == _normalize_for_metric("احمد")


def test_rouge_normalises_before_overlap():
    # Pre-fix this returned ~0; post-fix it should be high.
    score = rouge1_f1("النتيجة هي 2-1", "النتيجه هي 2-1")
    assert score > 0.8


def test_keyword_hit_full_match():
    assert keyword_hit(["الأهلي فاز على الزمالك"], ["الأهلي", "الزمالك"]) == 1.0


def test_keyword_hit_partial():
    assert keyword_hit(["الأهلي فاز"], ["الأهلي", "الزمالك"]) == pytest.approx(0.5)


def test_keyword_hit_empty_keywords_means_pass():
    assert keyword_hit(["anything"], []) == 1.0


def test_mrr_first_chunk_hit():
    assert mrr_score(["الأهلي 1-0", "خبر آخر"], ["الأهلي"]) == 1.0


def test_mrr_third_chunk_hit():
    assert mrr_score(["لا شيء", "لا شيء", "الأهلي 1-0"], ["الأهلي"]) == pytest.approx(1/3)


def test_mrr_no_hit():
    assert mrr_score(["a", "b"], ["لا يوجد"]) == 0.0


def test_intent_accuracy():
    assert intent_accuracy("lineup", "lineup") == 1.0
    assert intent_accuracy("lineup", "match_result") == 0.0


@pytest.mark.skipif(os.getenv("EVAL_FULL") != "1",
                    reason="loads 400MB model; set EVAL_FULL=1 to enable")
def test_embedding_similarity_close_strings():
    from evaluation.evaluate import embedding_similarity
    # Same meaning, different wording — should be high
    sim = embedding_similarity(
        "فاز الأهلي على الزمالك بنتيجة 2-1",
        "الأهلي تغلب على الزمالك 2-1",
    )
    assert sim > 0.85
