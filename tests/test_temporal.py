"""Unit tests for qa_engine/temporal.py date-window parsing."""

from __future__ import annotations

from datetime import date

from qa_engine.temporal import extract_date_window

TODAY = date(2026, 5, 10)


def test_yesterday_dialect_forms():
    for q in ("نتيجة مباراة الأهلي امبارح؟", "مين كسب أمس؟", "أهداف امس"):
        assert extract_date_window(q, TODAY) == ("2026-05-09", "2026-05-10"), q


def test_today_forms():
    for q in ("أخبار مران الأهلي النهاردة", "مباريات اليوم"):
        assert extract_date_window(q, TODAY) == ("2026-05-10", "2026-05-10"), q


def test_last_week():
    assert extract_date_window("نتائج الأسبوع الماضي", TODAY) == ("2026-05-03", "2026-05-10")
    assert extract_date_window("مباريات الاسبوع اللي فات", TODAY) == ("2026-05-03", "2026-05-10")


def test_explicit_arabic_date_gets_two_day_tail():
    # Match reports land up to 2 days after the fixture.
    assert extract_date_window("مباراة 15 مارس 2026", TODAY) == ("2026-03-15", "2026-03-17")


def test_explicit_arabic_date_defaults_to_current_year():
    assert extract_date_window("مباراة 15 مارس", TODAY) == ("2026-03-15", "2026-03-17")


def test_explicit_iso_date():
    assert extract_date_window("مباراة 2026-03-15", TODAY) == ("2026-03-15", "2026-03-17")


def test_no_date_signal_returns_nones():
    assert extract_date_window("من سجل هدف الأهلي؟", TODAY) == (None, None)
    assert extract_date_window("آخر أخبار محمد صلاح", TODAY) == (None, None)


def test_invalid_date_returns_nones():
    assert extract_date_window("مباراة 2026-13-99", TODAY) == (None, None)
