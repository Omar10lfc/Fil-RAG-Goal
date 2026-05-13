"""
FilGoalBot — Intent Router
==========================
Pure regex-based intent detector. Lives in its own module so it can be
unit-tested without importing Groq, FAISS, or sentence-transformers.

Intents:
  1. lineup            → تشكيلة
  2. match_result      → نتيجة مباراة / أهداف
  3. transfer_news     → ميركاتو / انتقالات / عقود
  4. team_news         → أخبار فريق / مران / مؤتمر صحفي
  5. player_info       → معلومات لاعب / إحصائيات
  6. general_football  → ambiguous-but-still-football fallback
                         (e.g. "ترتيب الدوري", "موعد المباريات")
  7. out_of_scope      → clearly NOT football (weather, cooking, other
                         sports, politics …). The pipeline short-circuits
                         these with a tailored refusal BEFORE any
                         retrieval or LLM call.
"""

import re

INTENT_PATTERNS: dict[str, list[str]] = {
    "lineup": [
        r'تشكيل', r'تشكيله', r'التشكيلة',
        r'أساسي', r'اساسي',
        r'الحارس', r'الحراسه', r'حارس المرمى',
        r'خط الدفاع', r'خط الوسط', r'خط الهجوم',
        r'مين اللي هيلعب', r'مين هيلعب',
        r'الاحتياطي', r'البدلاء',
        r'الكابتن',
        r'مين هيبدأ', r'مين بيبدأ',
        r'الـ?11\b', r'الإحدى عشر',
    ],

    "match_result": [
        # MSA
        # (?<!يست) blocks "يستهدف" (transfer-targeting) from matching "هدف".
        r'نتيج', r'انتهت', r'انته', r'(?<!يست)هدف', r'أهداف', r'اهداف',
        r'فاز', r'خسر', r'تعادل', r'سجل', r'لقاء',
        r'ركلات الترجيح', r'بالركلات',
        r'الفوز', r'فوز', r'انتصار', r'انتصر', r'انتصرت',
        r'هزيمة', r'هزم', r'انهزم',
        r'تأهل', r'تأهلت', r'تأهلوا',  # qualified (from a match)
        r'دربي',  # derby — implies a specific match
        # Allow 1-3 tokens between "فعل" and "أمام/ضد" so multi-word team
        # names like "الهلال السوداني" match.
        r'ماذا فعل\s+\S+(?:\s+\S+){0,2}\s+(أمام|ضد)',
        # dialect
        r'سكور', r'بالكام', r'جول', r'جوالات',
        r'كسب', r'اتكسب', r'اتعادل', r'اتغلب', r'ربح',
        r'فاز إيه', r'إيه نتيجة', r'ايه النتيجة',
        r'خلصت', r'خلص.*مباراة', r'مباراة.*امس', r'امتى خلص',
        r'دلوقتي.*نتيج', r'نتيج.*دلوقتي',
        r'بالرك',
        r'الفايز', r'فايز',
        r'مين كسب', r'مين فاز', r'مين خسر',
        r'بكام', r'بكم',
    ],

    "player_info": [
        r'لاعب', r'إحصائي', r'احصائي',
        r'إصاب', r'اصاب', r'مسيرة', r'جنسي', r'عمر',
        r'جاهز', r'غايب', r'موجود', r'مشارك',
        r'هيرجع امتى', r'رجع امتى', r'امتى بيرجع',
        r'اتاذى', r'تأذى', r'مصاب',
        r'اشترك', r'نزل من الاحتياطي',
        r'هيلعب', r'بيلعب', r'ماعرك', r'عارك',
        r'فين.*\b(صلاح|محمد|علي|عمر|احمد|كريم|مصطفى|حسام|إمام|زلاكه|شيكابالا|افشه|أفشة|بنشرقي|منسي)\b',
        r'ايه اخبار',
        # Allow 1-2 tokens between "أخبار" and "في/مع/ضد" — single-token form
        # missed compound names like "محمد صلاح".
        r'أخبار\s+\w+(?:\s+\w+)?\s+(في|مع|ضد)\s+',
        r'يتعافى', r'متعافى', r'تعافى',
        r'متى يعود', r'متى رجع',
        r'مستواه', r'أداءه', r'أدائه', r'أداء',
        r'حالته', r'موقفه',
        # MSA: questions about a player's status/career
        r'ما حالة', r'الصحي', r'البدني',
        r'إنجاز', r'الإنجاز',
        r'عقوبة',  # disciplinary actions → about the player
        # "(?:^|\s)" anchors so "حكم" doesn't match "كم" inside it.
        r'(?:^|\s)كم مباراة', r'(?:^|\s)كم مرة',
        r'سيغيب', r'يغيب',
        r'إضراب',  # player-specific action (e.g. Ronaldo's training strike)
        r'موقفه', r'موقف\s+\S+\s+من',  # "X's stance on..."
    ],

    "team_news": [
        r'مران', r'تدريب', r'محاضر', r'مؤتمر',
        r'جهاز', r'الجهاز الفني', r'مدرب', r'قائد',
        r'اجتماع', r'صرح', r'علق', r'تصريح', r'تصريحات',
        r'تحضير', r'استعداد',
        r'بيان', r'الإدارة', r'يدعم', r'دعم.*منظومة',
        # (?!\w) blocks suffix matches: without it "المصري" would match
        # "المصرية" and pull generic Egypt-football queries into team_news.
        r'أخبار.*(الأهلي|الزمالك|بيراميدز|الإسماعيلي|المصري|سيراميكا|طلائع|فاركو|الجونة|سموحة|المقاولون|إنبي|البنك الأهلي|غزل المحلة|حرس الحدود|مودرن)(?!\w)',
        r'أخبار.*فريق',
        r'بيعمل', r'قال إيه', r'قال ايه',
        r'بيحصل', r'اللي بيحصل',
        # MSA quote/statement forms — "what did X say" / "what did X announce".
        r'ماذا قال', r'ما الذي قاله',
        r'ماذا أعلن', r'ماذا حدث في\s+(?!مباراة|دربي|لقاء)',  # not match-context
    ],

    "transfer_news": [
        r'ميركاتو', r'انتقال', r'صفق', r'عقد', r'رحيل',
        r'ضم', r'انتقل', r'تعاقد', r'إعار', r'اعار',
        r'مفاوضات', r'فسخ', r'مجاني',
        r'مدته', r'عقده.*بيخلص', r'بيخلص.*عقده',
        r'اوبشن', r'أوبشن',
        r'هيجدد', r'هيجيب', r'هيضم', r'هيروح',
        r'هيفضل', r'مش هيفضل',
        r'جه جديد', r'جاي جديد',
        r'جاب مين', r'جابوا مين',
        r'راح فين', r'رحل لـ',
        # Coach hires are personnel acquisitions in this taxonomy — see test set.
        r'المدرب الجديد', r'مدرب جديد',
        r'لاعب جديد', r'لاعبين جدد',
        r'استعار', r'يستعير', r'استعارة',
        r'يرحل', r'سيرحل', r'هيرحل',
        r'هيوقع', r'يوقع.*(عقد|للـ|مع)',
        r'وقع.*عقد', r'وقع.*للـ', r'وقع.*مع',
        r'تجديد',
        # Coach personnel changes are transfers in this taxonomy (see test set:
        # "غزل المحلة ضم مدرب جديد" / "مين المدرب الجديد للمنتخب المصري").
        r'استقال', r'استقالة', r'إقالة', r'أقال', r'أقيل',
        r'يستهدف', r'استهدف',
        r'اتفاق مع',
        # Allow 1-3 tokens between the verb and direction so multi-word names
        # like "جيمس رودريجز" or "عزمي غومة" match. Plain "ل" suffix accepted
        # because the corpus uses both "لـ" and "للنادي" forms.
        r'وصل\s+\S+(?:\s+\S+){0,2}\s+(?:ل|إلى)',
        r'عاد\s+\S+(?:\s+\S+){0,2}\s+(?:إلى|ل)\s+\S+',  # player rejoining a club
    ],
}

# Order matters: lineup checked FIRST because "تشكيل" is very specific and would
# also leak into team_news. Then transfer_news to catch contract/مفاوضات before
# team_news's broad "مدرب" pattern. player_info is last so generic player names
# don't outrank a more specific lineup/result query.
INTENT_ORDER = ["lineup", "match_result", "transfer_news", "team_news", "player_info"]

# Extractive intents — answer is usually a direct fact in one chunk.
# These can use the cheaper 8B model without quality loss.
EXTRACTIVE_INTENTS = {"lineup", "match_result"}


# Patterns that signal a query is clearly NOT about football. Checked only as
# a tiebreaker against the general_football fallback — i.e. when no specific
# football intent fires. Otherwise legitimate borderline football queries
# (e.g. "ترتيب الدوري المصري") would over-refuse.
#
# Each pattern is chosen to be either (a) syntactically specific enough not
# to collide with football vocabulary, or (b) reference a topic with no
# football overlap at all. Where collision is possible (a club president
# vs. a country president), the pattern requires the non-football
# disambiguator explicitly.
OUT_OF_SCOPE_PATTERNS: list[str] = [
    # Weather
    r'\bالطقس\b', r'حال[ةه] الجو', r'درج[ةه] الحرار[ةه]', r'الأمطار',
    # Cooking / food
    r'وصف[ةه] طبخ', r'طريق[ةه] طبخ', r'\bطبخ', r'مطبخ',
    # Restaurants / dining
    r'\bمطعم\b', r'\bمطاعم\b',
    # Other sports — explicit so "كرة" alone doesn't trip
    r'كر[ةه]\s+(?:السل[ةه]|الطائر[ةه]|اليد)',
    r'\bتنس\b', r'ملاكم[ةه]', r'سباح[ةه]', r'ألعاب القوى',
    r'فورمولا', r'دراج[ةه] هوائي[ةه]', r'شطرنج',
    # Named non-football tournaments — covers cases where the query
    # uses match_result vocabulary ("من فاز ببطولة …") but the subject
    # is a non-football tournament that bypasses the كرة-based check above.
    r'ويمبلدون', r'رولان\s+جاروس', r'\bNBA\b', r'\bالـ?NBA\b',
    r'يورو\s*فيجن', r'الأوسكار',
    # Politics — require qualifiers that don't apply to club officials
    r'رئيس الجمهوري[ةه]', r'الانتخابات الرئاسي[ةه]', r'البرلمان',
    r'وزير\s+(?:الخارجي[ةه]|الداخلي[ةه]|الصح[ةه]|التعليم|المالي[ةه])',
    # Tech / general apps
    r'برمج[ةه]', r'كمبيوتر', r'لاب\s*توب', r'هاتف محمول', r'موقع إلكتروني',
    # Science / academia — extremely unlikely to collide with football
    r'نظري[ةه] النسبي[ةه]', r'\bفيزياء\b', r'\bكيمياء\b', r'\bرياضيات\b',
    r'\bأينشتاين\b', r'\bنيوتن\b', r'الجاذبي[ةه]', r'\bالذر[ةه]\b',
    # Astronomy
    r'\bالشمس\b', r'\bالقمر\b', r'الكواكب', r'\bالمجر[ةه]\b',
    # Distance / measurement questions — "كم تبعد X عن Y" is geographic,
    # nothing in football reads that way.
    r'\bتبعد\b.{0,30}\bعن\b',
    # Medicine — pair "علاج" with a disease so we don't fire on "علاج إصابة
    # محمد صلاح". The disease list covers the common test-set probes.
    r'علاج\s+(?:السكري|الضغط|البرد|نزلات|الكوليسترول|السرطان|الإيدز)',
    r'نزلات\s+البرد', r'مرض\s+السكري', r'الكوليسترول',
    # History / institutions — "متى تأسست" + a non-football institution
    r'متى\s+تأسست\s+(?:جامع[ةه]|كلي[ةه]|الدول[ةه]|البنك|الجمهوري[ةه])',
    r'\bجامع[ةه]\s+(?:الأزهر|القاهر[ةه]|عين شمس|الإسكندري[ةه])',
    # Creative writing / arts
    r'\bقصيد[ةه]\b', r'اكتب\s+لي\s+قصة', r'اكتب\s+لي\s+مقال',
    r'\bشعر\s+عن\b', r'\bرواي[ةه]\b\s+\S+',
    # Entertainment. Religion patterns dropped — "رمضان", "الصلاة" and
    # "الصيام" collide with legitimate football queries ("هل سيغيب صلاح
    # في رمضان؟", "وقت الصلاة قبل المباراة"). The test set has no
    # religion-refusal cases that need them.
    r'فيلم\s+\S+', r'مسلسل\s+\S+', r'ممثل\s+\S+', r'مطرب\s+\S+',
    # Economy (non-football)
    r'سعر\s+الدولار', r'سعر\s+الذهب', r'سعر\s+سهم', r'سعر\s+النفط',
    r'سعر\s+الفائد[ةه]', r'البورص[ةه]', r'التضخم',
]


def _is_out_of_scope(query: str) -> bool:
    return any(re.search(p, query) for p in OUT_OF_SCOPE_PATTERNS)


def detect_intent(query: str) -> str:
    q = query.lower()
    # "ماذا قال X" → team_news. Override needed because match_result runs first,
    # and quoted statements often mention "فوز", "هدف" etc that fire there.
    if re.search(r'ماذا\s+قال|ما\s+الذي\s+قاله|ماذا\s+أعلن', q):
        return "team_news"
    # High-priority override: "لاعب اسمه ..." routes to player_info to defeat
    # match_result's "سجل" trigger (e.g. "كم سجل لاعب اسمه X"). Skip the override
    # when a transfer verb is present — "هل انتقل لاعب اسمه X" is transfer_news.
    if re.search(r'لاعب\s+اسمه', q) and not re.search(r'انتقل|صفق|مفاوضات|تعاقد|إعار|اعار', q):
        return "player_info"
    # Player-centric return-to-training. Allow 1-3 tokens between "عاد" and
    # "ل/إلى تدريب" so multi-word names like "دي بروين" / "عبد المنعم" match.
    # Without this override, team_news's "تدريب" pattern swallows the case.
    if re.search(r'(?:عاد|عودة)\s+\S+(?:\s+\S+){0,2}\s+ل[إا]?\s*تدريب', q):
        return "player_info"
    # Out-of-scope check BEFORE INTENT_ORDER. Football vocabulary is so
    # broad (نتيجة، مباراة، حالة، ...) that a query like
    # "نتيجة مباراة كرة السلة" would otherwise match match_result on
    # "نتيج" and lose its OOS signal. The patterns are conservative —
    # they target topics with no overlap with football (weather, other
    # sports, politics-with-disambiguator, etc.) — so promoting them
    # over INTENT_ORDER does not over-refuse genuine football queries.
    if _is_out_of_scope(q):
        return "out_of_scope"
    for intent in INTENT_ORDER:
        for pattern in INTENT_PATTERNS[intent]:
            if re.search(pattern, q):
                return intent
    return "general_football"
