"""Unit tests for the intent router. Pure regex, no model loads."""

import pytest

from qa_engine.intent import EXTRACTIVE_INTENTS, detect_intent


@pytest.mark.parametrize("query,expected", [
    # lineup — must beat team_news/match_result for words like "تشكيل"
    ("ما تشكيل الأهلي قبل مباراة الترجي؟",                "lineup"),
    ("مين حارس مرمى الأهلي الأساسي؟",                     "lineup"),
    ("إيه التشكيلة الأساسية للزمالك؟",                    "lineup"),

    # match_result — MSA + dialect
    ("ما نتيجة مباراة بيراميدز والجيش الملكي؟",            "match_result"),
    ("فاز إيه مانشستر سيتي؟",                              "match_result"),
    ("من سجل أهداف الأهلي في آخر مباراة؟",                 "match_result"),
    ("نتيجة الكلاسيكو الأخير برشلونة وريال مدريد؟",        "match_result"),

    # transfer_news — must beat team_news for "صفقة" / "هيجدد"
    ("ما آخر صفقات الزمالك في ميركاتو؟",                  "transfer_news"),
    ("صلاح هيجدد مع ليفربول؟",                             "transfer_news"),
    ("أرسنال هيجدد عقد رايس وتيمبر؟",                      "transfer_news"),

    # team_news
    ("إيه أخبار مران الأهلي النهارده؟",                    "team_news"),
    ("كاريك قال إيه في مؤتمر الصحفي؟",                     "team_news"),

    # player_info
    ("هل الونش جاهز للعب؟",                                "player_info"),
    ("إيه إصابة تروسارد؟",                                 "player_info"),

    # general fallback — ambiguous but still football
    ("ترتيب الدوري المصري الحالي؟",                        "general_football"),
    ("موعد مباريات الأسبوع القادم؟",                       "general_football"),

    # out_of_scope — clearly NOT football. Must NOT fall into general_football.
    ("ما حالة الطقس اليوم؟",                               "out_of_scope"),
    ("اعطني وصفة طبخ المحشي",                              "out_of_scope"),
    ("من رئيس الجمهورية الحالي؟",                          "out_of_scope"),
    ("نتيجة مباراة كرة السلة بين الأهلي والزمالك",          "out_of_scope"),
    ("سعر الدولار اليوم",                                  "out_of_scope"),
])
def test_detect_intent(query: str, expected: str):
    assert detect_intent(query) == expected


def test_extractive_intents_are_fact_lookups():
    """Sanity check: only intents whose answer is a direct fact in one chunk
    should be cheap-modeled. Reasoning-heavy intents should NOT be in this set."""
    assert "lineup" in EXTRACTIVE_INTENTS
    assert "match_result" in EXTRACTIVE_INTENTS
    assert "team_news" not in EXTRACTIVE_INTENTS
    assert "general_football" not in EXTRACTIVE_INTENTS


def test_unknown_query_falls_through():
    assert detect_intent("سؤال عشوائي بدون كلمات مفتاحية") == "general_football"
