"""
FilGoalBot — temporal query parsing for match_result date-window filtering.

match_result is the weakest intent (Kw-hit 0.785): questions like "نتيجة
مباراة الأهلي امبارح؟" need articles from a specific day, but the recency
boost alone is too gentle to exclude older same-fixture coverage. Wired in
qa_engine/rag_pipeline._prepare() for match_result only.

Windows (all inclusive, ISO YYYY-MM-DD):
    امبارح / أمس      → [d−1, d]
    النهاردة / اليوم  → [d, d]
    الأسبوع الماضي    → [d−7, d]
    explicit dates    → [d, d+2]  (match reports land up to 2 days later)
"""

from __future__ import annotations

import re
from datetime import date, timedelta

_AR_MONTHS = {
    "يناير": 1, "فبراير": 2, "مارس": 3, "أبريل": 4, "ابريل": 4,
    "مايو": 5, "يونيو": 6, "يوليو": 7, "أغسطس": 8, "اغسطس": 8,
    "سبتمبر": 9, "أكتوبر": 10, "اكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12,
}

_YESTERDAY = re.compile(r'امبارح|إمبارح|البارح|أمس|امس')
_TODAY = re.compile(r'النهارده|النهاردة|اليوم')
_LAST_WEEK = re.compile(r'الأسبوع\s+الماضي|الاسبوع\s+الماضي|الأسبوع\s+اللي\s+فات|الاسبوع\s+اللي\s+فات')
_EXPLICIT_AR = re.compile(
    r'(\d{1,2})\s+(يناير|فبراير|مارس|أبريل|ابريل|مايو|يونيو|يوليو|'
    r'أغسطس|اغسطس|سبتمبر|أكتوبر|اكتوبر|نوفمبر|ديسمبر)(?:\s+(\d{4}))?'
)
_EXPLICIT_ISO = re.compile(r'(\d{4})-(\d{1,2})-(\d{1,2})')


def _iso(d: date) -> str:
    return d.isoformat()


def extract_date_window(
    query: str,
    today: date | None = None,
) -> tuple[str | None, str | None]:
    """Parse an Arabic time reference into an inclusive (from, to) ISO window.

    Returns (None, None) when the query carries no date signal — callers then
    skip filtering entirely. `today` is injectable for tests.
    """
    today = today or date.today()
    q = query

    if _LAST_WEEK.search(q):
        return _iso(today - timedelta(days=7)), _iso(today)
    if _YESTERDAY.search(q):
        return _iso(today - timedelta(days=1)), _iso(today)
    if _TODAY.search(q):
        return _iso(today), _iso(today)

    m = _EXPLICIT_AR.search(q)
    if m:
        try:
            day = int(m.group(1))
            month = _AR_MONTHS[m.group(2)]
            year = int(m.group(3)) if m.group(3) else today.year
            d = date(year, month, day)
        except ValueError:
            return None, None
        return _iso(d), _iso(d + timedelta(days=2))

    m = _EXPLICIT_ISO.search(q)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None, None
        return _iso(d), _iso(d + timedelta(days=2))

    return None, None
