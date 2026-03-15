# """
# FilGoalBot RAG Pipeline
# ========================
# Intents:
#   1. match_result       → نتيجة مباراة / أهداف
#   2. lineup             → تشكيلة
#   3. player_info        → معلومات لاعب / إحصائيات
#   4. team_news          → أخبار فريق / مران / مؤتمر صحفي
#   5. transfer_news      → ميركاتو / انتقالات / عقود
#   6. general_football   → سؤال عام عن كرة القدم
# """

# import os
# import logging
# import re
# from groq import Groq
# from dotenv import load_dotenv
# from retrieval.hybrid_retriever import FilGoalRetriever
# load_dotenv()
# log = logging.getLogger("rag")

# GROQ_MODEL  = "llama-3.3-70b-versatile"
# MAX_TOKENS  = 600
# TEMPERATURE = 0.2
# TOP_K       = 6


# # ─── Intent Router ────────────────────────────────────────────────────────────

# INTENT_PATTERNS = {
#     "lineup": [
#         r'تشكيل', r'تشكيله', r'أساسي', r'اساسي', r'الحارس', r'التشكيلة',
#     ],
#     "match_result": [
#         r'نتيج', r'انته', r'هدف', r'اهداف', r'فاز', r'خسر', r'تعادل',
#         r'سجل', r'مباراة.*امس', r'لقاء', r'سكور',
#     ],
#     "player_info": [
#         r'لاعب', r'إحصائي', r'احصائي', r'إصاب', r'اصاب', r'عمر',
#         r'جنسي', r'مسيرة',
#     ],
#     "team_news": [
#         r'مران', r'تدريب', r'محاضر', r'مؤتمر', r'جهاز', r'مدرب', r'قائد',
#         r'أخبار.*فريق', r'أخبار.*الأهلي', r'أخبار.*الزمالك',
#     ],
#     "transfer_news": [
#         r'ميركاتو', r'انتقال', r'صفق', r'عقد', r'رحيل', r'ضم',
#         r'انتقل', r'تعاقد', r'إعار', r'اعار',
#     ],
# }

# # lineup checked FIRST — "تشكيل" is very specific and would also match team_news otherwise
# INTENT_ORDER = ["lineup", "match_result", "transfer_news", "team_news", "player_info"]

# def detect_intent(query: str) -> str:
#     q = query.lower()
#     for intent in INTENT_ORDER:
#         for pattern in INTENT_PATTERNS[intent]:
#             if re.search(pattern, q):
#                 return intent
#     return "general_football"


# # ─── System Prompts ───────────────────────────────────────────────────────────

# _BASE = """\
# أنت FilGoalBot — مساعد ذكي متخصص في أخبار كرة القدم المصرية والعربية والعالمية.
# تجيب بالعربية فقط، بإيجاز ودقة، مستنداً إلى المعلومات المقدمة لك فقط.
# إذا لم تجد المعلومة في السياق، قل بوضوح: "لا تتوفر لديّ هذه المعلومة حالياً."
# لا تخترع نتائج أو أهدافاً أو صفقات غير موجودة في المصادر.\
# """

# INTENT_PROMPTS = {
#     "lineup":           _BASE + "\nاذكر التشكيلة بوضوح: الحارس، الدفاع، الوسط، الهجوم.",
#     "match_result":     _BASE + "\nركز على النتيجة النهائية، الهدافين، وأبرز أحداث المباراة.",
#     "player_info":      _BASE + "\nركز على معلومات اللاعب: الأداء، الإحصائيات، الإصابات.",
#     "team_news":        _BASE + "\nركز على آخر أخبار الفريق: المران، التصريحات، المؤتمرات.",
#     "transfer_news":    _BASE + "\nركز على تفاصيل الصفقة: الفريقين، القيمة، مدة العقد.",
#     "general_football": _BASE,
# }

# # Maps intent → article_type filter passed to retriever.
# # lineup and match_result map to specific article types in the pipeline.
# # team_news covers both training AND press_conference — no filter, let RRF decide.
# # player_info and general_football — no filter needed.
# FILTER_MAP = {
#     "lineup":        {"filter_type": "lineup"},
#     "match_result":  {"filter_type": "match_result"},
#     "transfer_news": {"filter_type": "transfer"},
# }


# # ─── Context builder ──────────────────────────────────────────────────────────

# def _build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
#     """
#     Build LLM context string and sources list from retrieved chunks.
#     Uses body_clean (readable Arabic) — NOT text which is the normalised
#     embedding string (alef variants stripped, ة→ه) and would produce garbled answers.
#     Caps each chunk at 500 chars to stay within Groq token limits.
#     """
#     context_parts = []
#     sources = []

#     for i, chunk in enumerate(chunks, 1):
#         title = chunk.get("title", "")
#         date  = chunk.get("pub_date", "")[:10]

#         body = chunk.get("body_clean") or chunk.get("text", "")
#         body = body[:500]

#         header = f"[{i}] {title}"
#         if date:
#             header += f" ({date})"

#         context_parts.append(f"{header}\n{body}")
#         sources.append({
#             "title":        title,
#             "url":          chunk.get("source_url", ""),
#             "pub_date":     date,
#             "article_type": chunk.get("article_type", ""),
#             "league":       chunk.get("league", ""),
#         })

#     return "\n\n---\n\n".join(context_parts), sources


# # ─── RAG Pipeline ─────────────────────────────────────────────────────────────

# class FilGoalRAG:
#     def __init__(self):
#         self.retriever = FilGoalRetriever()
#         self.groq      = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

#     def load(self):
#         self.retriever.load()
#         log.info("✅ FilGoalRAG ready")

#     def answer(self, query: str) -> dict:
#         """
#         Full RAG: retrieve → build context → call Groq → return answer + metadata.

#         Returns:
#             {
#                 "answer":  str,           Arabic answer
#                 "intent":  str,           detected intent
#                 "sources": list[dict],    top-3 source articles
#             }
#         """
#         intent  = detect_intent(query)
#         filters = FILTER_MAP.get(intent, {})
#         log.info(f"Intent: {intent} | Filters: {filters} | Query: {query}")

#         # ── 1. Retrieve ───────────────────────────────────────────────────────
#         chunks = self.retriever.retrieve(query, top_k=TOP_K, **filters)

#         if not chunks:
#             return {
#                 "answer":  "لا تتوفر لديّ معلومات كافية للإجابة على هذا السؤال.",
#                 "intent":  intent,
#                 "sources": [],
#             }

#         # ── 2. Build context ──────────────────────────────────────────────────
#         context, sources = _build_context(chunks)

#         # ── 3. Call Groq ──────────────────────────────────────────────────────
#         system_prompt = INTENT_PROMPTS[intent]
#         user_prompt   = f"السياق:\n\n{context}\n\n{'─'*40}\n\nالسؤال: {query}"

#         try:
#             response = self.groq.chat.completions.create(
#                 model=GROQ_MODEL,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user",   "content": user_prompt},
#                 ],
#                 max_tokens=MAX_TOKENS,
#                 temperature=TEMPERATURE,
#             )
#             answer = response.choices[0].message.content.strip()
#         except Exception as e:
#             log.error(f"Groq error: {e}")
#             answer = "حدث خطأ أثناء توليد الإجابة، يرجى المحاولة مرة أخرى."

#         return {
#             "answer":  answer,
#             "intent":  intent,
#             "sources": sources[:3],
#         }


# # ─── CLI smoke-test ───────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     import sys
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

#     bot = FilGoalRAG()
#     bot.load()

#     test_questions = [
#         "ما نتيجة مباراة بيراميدز والجيش الملكي؟",
#         "ما تشكيل الأهلي قبل مباراة الترجي؟",
#         "ما آخر أخبار صلاح في ليفربول؟",
#         "من سجل لطلائع الجيش في كأس مصر؟",
#         "ما آخر صفقات الزمالك في ميركاتو؟",
#     ]

#     questions = sys.argv[1:] if len(sys.argv) > 1 else test_questions

#     for q in questions:
#         print(f"\n{'='*60}")
#         print(f" {q}")
#         result = bot.answer(q)
#         print(f"Intent : {result['intent']}")
#         print(f"Answer : {result['answer']}")
#         print(f"Sources: {[s['title'][:45] for s in result['sources']]}")
"""
FilGoalBot RAG Pipeline
========================
Intents:
  1. match_result       → نتيجة مباراة / أهداف
  2. lineup             → تشكيلة
  3. player_info        → معلومات لاعب / إحصائيات
  4. team_news          → أخبار فريق / مران / مؤتمر صحفي
  5. transfer_news      → ميركاتو / انتقالات / عقود
  6. general_football   → سؤال عام عن كرة القدم
"""

import os
import logging
import re

from dotenv import load_dotenv
from groq import Groq

from retrieval.hybrid_retriever import FilGoalRetriever

# ── Logging configured FIRST so any import-time errors are visible ────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("rag")

# ── Load .env before anything reads os.getenv ─────────────────────────────────
load_dotenv()

GROQ_MODEL  = "llama-3.3-70b-versatile"
MAX_TOKENS  = 600
TEMPERATURE = 0.2
TOP_K       = 6


# ─── Intent Router ────────────────────────────────────────────────────────────

INTENT_PATTERNS = {
    "lineup": [                          # ← this key was missing
        r'تشكيل', r'تشكيله', r'أساسي', r'اساسي', r'الحارس', r'التشكيلة',
    ],
    "player_info": [
        r'لاعب', r'إحصائي', r'احصائي', r'إصاب', r'اصاب', r'عمر',
        r'جنسي', r'مسيرة',
        r'جاهز', r'هيلعب', r'بيلعب', r'ماعرك', r'عارك',
        r'أخبار\s+\w+\s+في\s+',
    ],
    "match_result": [
        r'نتيج', r'انته', r'هدف', r'اهداف', r'فاز', r'خسر', r'تعادل',
        r'سجل', r'مباراة.*امس', r'لقاء', r'سكور',
        r'فاز إيه', r'إيه نتيجة', r'كسب', r'اتكسب', r'اتعادل',
    ],
    "team_news": [
        r'مران', r'تدريب', r'محاضر', r'مؤتمر', r'جهاز', r'مدرب', r'قائد',
        r'أخبار.*فريق', r'أخبار.*الأهلي', r'أخبار.*الزمالك',
        r'بيعمل', r'بيحصل', r'اللي بيحصل',
    ],
    "transfer_news": [
        r'ميركاتو', r'انتقال', r'صفق', r'عقد', r'رحيل', r'ضم',
        r'انتقل', r'تعاقد', r'إعار', r'اعار',
        r'هيجدد', r'هيجيب', r'جه جديد', r'جاي جديد', r'هيضم', r'هيروح',
    ],
}
# lineup checked FIRST — "تشكيل" is very specific
INTENT_ORDER = ["lineup", "match_result", "transfer_news", "team_news", "player_info"]

def detect_intent(query: str) -> str:
    q = query.lower()
    for intent in INTENT_ORDER:
        for pattern in INTENT_PATTERNS[intent]:
            if re.search(pattern, q):
                return intent
    return "general_football"


# ─── System Prompts ───────────────────────────────────────────────────────────

_BASE = """\
أنت FilGoalBot — مساعد ذكي متخصص في أخبار كرة القدم المصرية والعربية والعالمية.
تجيب بالعربية فقط، بإيجاز ودقة، مستنداً إلى المعلومات المقدمة لك فقط.
إذا لم تجد المعلومة في السياق، قل بوضوح: "لا تتوفر لديّ هذه المعلومة حالياً."
لا تخترع نتائج أو أهدافاً أو صفقات غير موجودة في المصادر.\
"""

INTENT_PROMPTS = {
    "lineup":           _BASE + "\nاذكر التشكيلة بوضوح: الحارس، الدفاع، الوسط، الهجوم.",
    "match_result":     _BASE + "\nركز على النتيجة النهائية، الهدافين، وأبرز أحداث المباراة.",
    "player_info":      _BASE + "\nركز على معلومات اللاعب: الأداء، الإحصائيات، الإصابات.",
    "team_news":        _BASE + "\nركز على آخر أخبار الفريق: المران، التصريحات، المؤتمرات.",
    "transfer_news":    _BASE + "\nركز على تفاصيل الصفقة: الفريقين، القيمة، مدة العقد.",
    "general_football": _BASE,
}

FILTER_MAP = {
    "lineup":        {"filter_type": "lineup"},
    "match_result":  {"filter_type": "match_result"},
    "transfer_news": {"filter_type": "transfer"},
}


# ─── Context builder ──────────────────────────────────────────────────────────

def _build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    context_parts = []
    sources = []

    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("title", "")
        date  = chunk.get("pub_date", "")[:10]

        body = chunk.get("body_clean") or chunk.get("text", "")
        body = body[:500]

        header = f"[{i}] {title}"
        if date:
            header += f" ({date})"

        context_parts.append(f"{header}\n{body}")
        sources.append({
            "title":        title,
            "url":          chunk.get("source_url", ""),
            "pub_date":     date,
            "article_type": chunk.get("article_type", ""),
            "league":       chunk.get("league", ""),
        })

    return "\n\n---\n\n".join(context_parts), sources


# ─── RAG Pipeline ─────────────────────────────────────────────────────────────

class FilGoalRAG:
    def __init__(self):
        self.retriever = FilGoalRetriever()
        self.groq      = None   # lazy — initialised in load() after .env is confirmed

    def load(self):
        # ── Confirm API key exists before doing the heavy model load ──────────
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set.\n"
                "Add it to your .env file:\n"
                "  GROQ_API_KEY=gsk_..."
            )
        self.groq = Groq(api_key=api_key)
        log.info("✅ Groq client ready")

        self.retriever.load()
        log.info("FilGoalRAG ready")

    def answer(self, query: str,
           filter_type: str | None = None,
           filter_league: str | None = None,
           filter_team: str | None = None) -> dict:
        intent  = detect_intent(query)
        filters = FILTER_MAP.get(intent, {})
        # Override with explicit filters if provided
        if filter_type is not None:
            filters["filter_type"] = filter_type
        if filter_league is not None:
            filters["filter_league"] = filter_league
        if filter_team is not None:
            filters["filter_team"] = filter_team
        log.info(f"Intent: {intent} | Filters: {filters} | Query: {query}")

        # ── 1. Retrieve ───────────────────────────────────────────────────────
        chunks = self.retriever.retrieve(query, top_k=TOP_K, **filters)

        if not chunks:
            return {
                "answer":  "لا تتوفر لديّ معلومات كافية للإجابة على هذا السؤال.",
                "intent":  intent,
                "sources": [],
            }

        # ── 2. Build context ──────────────────────────────────────────────────
        context, sources = _build_context(chunks)

        # ── 3. Call Groq ──────────────────────────────────────────────────────
        system_prompt = INTENT_PROMPTS[intent]
        user_prompt   = f"السياق:\n\n{context}\n\n{'─'*40}\n\nالسؤال: {query}"

        try:
            response = self.groq.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            log.error(f"Groq error: {e}")
            answer = "حدث خطأ أثناء توليد الإجابة، يرجى المحاولة مرة أخرى."

        return {
            "answer":  answer,
            "intent":  intent,
            "sources": sources[:3],
        }


# ─── CLI smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    try:
        bot = FilGoalRAG()
        bot.load()
    except Exception as e:
        log.error(f"❌ Startup failed: {e}")
        sys.exit(1)

    test_questions = [
        "ما نتيجة مباراة بيراميدز والجيش الملكي؟",
        "ما تشكيل الأهلي قبل مباراة الترجي؟",
        "ما آخر أخبار صلاح في ليفربول؟",
        "من سجل لطلائع الجيش في كأس مصر؟",
        "ما آخر صفقات الزمالك في ميركاتو؟",
    ]

    questions = sys.argv[1:] if len(sys.argv) > 1 else test_questions

    for q in questions:
        print(f"\n{'='*60}")
        print(f" {q}")
        try:
            result = bot.answer(q)
            print(f"Intent : {result['intent']}")
            print(f"Answer : {result['answer']}")
            print(f"Sources: {[s['title'][:45] for s in result['sources']]}")
        except Exception as e:
            log.error(f"❌ Question failed: {e}")