"""Regression tests for the trailing `[N]` template-leak strip.

Observed on the live HF Space: a multi-source 70B answer ended with a
literal "[N]" — the model parroted the prompt's placeholder syntax
instead of substituting a real source number. PROMPT_VERSION was bumped
to 3 with explicit "never use the letter N" instructions, and a
post-processor in rag_pipeline strips any trailing `[N]`/`[n]` as
defence in depth. These tests lock that behaviour in.
"""

from __future__ import annotations

from qa_engine.rag_pipeline import _strip_template_leaks


def test_strips_exact_observed_leak():
    """The live example: long answer ending with a stray [N] token."""
    leaked = (
        "محمد صلاح أنهى صيامه التهديفي [1]. ويفكر صلاح في الاستمرار مع "
        "ليفربول [3]. كما وصل لهدفه رقم 8 في الموسم الجاري [4]. [N]"
    )
    cleaned = _strip_template_leaks(leaked)
    assert cleaned.endswith("[4].")
    assert "[N]" not in cleaned


def test_strips_lowercase_variant():
    assert _strip_template_leaks("text [1]. [n]") == "text [1]."


def test_strips_whitespace_padded_variant():
    # Models sometimes emit "[ N ]" or trailing newlines/spaces.
    assert _strip_template_leaks("text [1]. [ N ]") == "text [1]."
    assert _strip_template_leaks("text [1].\n[N]\n  ") == "text [1]."


def test_preserves_real_citations():
    """The strip must only catch the *trailing* [N]/[n] sentinel — never
    legitimate citation markers like [1] / [12] / [99]."""
    good = "تشكيل الأهلي: محمد الشناوي [1]، خط الدفاع [2]، خط الوسط [12]."
    assert _strip_template_leaks(good) == good


def test_does_not_strip_mid_string_n_token():
    """A [N] that isn't at the very end of the answer is not the bug we're
    targeting — leave it alone rather than mangle the structure."""
    text = "[N] is a placeholder pattern used in the prompt."
    assert _strip_template_leaks(text) == text


def test_handles_empty_and_whitespace_input():
    assert _strip_template_leaks("") == ""
    assert _strip_template_leaks("   ") == ""
    assert _strip_template_leaks("[N]") == ""
