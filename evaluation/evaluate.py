"""
FilGoalBot — Evaluation Suite
================================
Runs 3 ablation experiments + full end-to-end RAG evaluation.

Metrics:
  - ROUGE-1       → retrieval coverage (no external deps)
  - Hit Rate      → was at least one relevant chunk retrieved?
  - Intent Accuracy → did we detect the right intent?
  - MRR           → Mean Reciprocal Rank (ranking quality)
  - Latency (ms)  → retrieval speed

Experiments (Ablation):
  1. BM25 only       (sparse baseline)
  2. Dense only      (FAISS/embedding)
  3. Hybrid BM25+FAISS (RRF) ← expected winner

Per-intent breakdown:
  - match_result, player_info, team_news, transfer_news, general_football

Run:
    python -m evaluation.evaluate
    python -m evaluation.evaluate --rag       # full RAG eval (needs Groq)
    python -m evaluation.evaluate --save-report
"""

import os
import re
import json
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("evaluation")

EVAL_DIR       = Path("evaluation")
TEST_SET_PATH  = EVAL_DIR / "test_set.json"
REPORT_PATH    = EVAL_DIR / "report.json"


# ════════════════════════════════════════════════════════════════════════════
# Test Set — 50 Arabic football questions
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_TEST_SET = [

    # ── Match Results (10) ──────────────────────────────────────────────────
    {
        "question":   "ما نتيجة مباراة بيراميدز والجيش الملكي في دوري أبطال إفريقيا؟",
        "reference":  "تعادل بيراميدز مع الجيش الملكي في دوري أبطال إفريقيا",
        "intent":     "match_result",
        "difficulty": "easy",
        "keywords":   ["بيراميدز", "الجيش الملكي", "تعادل", "إفريقيا"],
    },
    {
        "question":   "من سجل أهداف الأهلي في آخر مباراة؟",
        "reference":  "سجل لاعبو الأهلي أهداف في المباراة",
        "intent":     "match_result",
        "difficulty": "medium",
        "keywords":   ["الأهلي", "هدف", "سجل"],
    },
    {
        "question":   "نتيجة مباراة طلائع الجيش وسيراميكا في كأس مصر؟",
        "reference":  "تأهل طلائع الجيش لنصف نهائي كأس مصر على حساب سيراميكا بركلات الترجيح",
        "intent":     "match_result",
        "difficulty": "easy",
        "keywords":   ["طلائع الجيش", "سيراميكا", "كأس مصر", "نصف النهائي"],
    },
    {
        "question":   "هل انتهت مباراة الأهلي والترجي؟",
        "reference":  "مباراة الأهلي والترجي في دوري أبطال إفريقيا",
        "intent":     "match_result",
        "difficulty": "medium",
        "keywords":   ["الأهلي", "الترجي"],
    },
    {
        "question":   "ما أهداف مباراة الزمالك وأوتوهو؟",
        "reference":  "مباراة الزمالك وأوتوهو في دوري أبطال إفريقيا",
        "intent":     "match_result",
        "difficulty": "hard",
        "keywords":   ["الزمالك", "أوتوهو"],
    },
    {
        "question":   "فاز إيه مانشستر سيتي؟",
        "reference":  "نتيجة مباراة مانشستر سيتي",
        "intent":     "match_result",
        "difficulty": "medium",
        "keywords":   ["مانشستر سيتي"],
    },
    {
        "question":   "من الفايز في الدوري الإنجليزي امبارح؟",
        "reference":  "نتائج الدوري الإنجليزي الممتاز",
        "intent":     "match_result",
        "difficulty": "hard",
        "keywords":   ["الدوري الإنجليزي"],
    },
    {
        "question":   "نتيجة الكلاسيكو الأخير برشلونة وريال مدريد؟",
        "reference":  "مباراة برشلونة وريال مدريد",
        "intent":     "match_result",
        "difficulty": "medium",
        "keywords":   ["برشلونة", "ريال مدريد"],
    },
    {
        "question":   "أهداف الجولة الأخيرة من الدوري المصري",
        "reference":  "أهداف الجولة الأخيرة في الدوري المصري",
        "intent":     "match_result",
        "difficulty": "hard",
        "keywords":   ["الدوري المصري", "أهداف"],
    },
    {
        "question":   "إيه نتيجة القادسية وأهلي جدة؟",
        "reference":  "نتيجة مباراة القادسية وأهلي جدة في الدوري السعودي",
        "intent":     "match_result",
        "difficulty": "hard",
        "keywords":   ["القادسية", "أهلي جدة"],
    },

    # ── Player Info (10) ──────────────────────────────────────────────────
    {
        "question":   "آخر أخبار محمد صلاح في ليفربول",
        "reference":  "محمد صلاح لاعب ليفربول المصري",
        "intent":     "player_info",
        "difficulty": "easy",
        "keywords":   ["صلاح", "ليفربول"],
    },
    {
        "question":   "هل الونش جاهز للعب؟",
        "reference":  "الونش لاعب الأهلي يؤدي مرحلة تأهيلية",
        "intent":     "player_info",
        "difficulty": "easy",
        "keywords":   ["الونش", "تأهيلي"],
    },
    {
        "question":   "ماعرك بن وايت في أرسنال؟",
        "reference":  "بن وايت لاعب أرسنال جاهز للعب",
        "intent":     "player_info",
        "difficulty": "medium",
        "keywords":   ["بن وايت", "أرسنال"],
    },
    {
        "question":   "إيه أخبار ماونت في مانشستر يونايتد؟",
        "reference":  "ماونت لاعب مانشستر يونايتد عائد من الإصابة",
        "intent":     "player_info",
        "difficulty": "medium",
        "keywords":   ["ماونت", "مانشستر يونايتد"],
    },
    {
        "question":   "مين حارس مرمى الأهلي الأساسي؟",
        "reference":  "حارس مرمى الأهلي",
        "intent":     "player_info",
        "difficulty": "medium",
        "keywords":   ["الأهلي", "حارس"],
    },
    {
        "question":   "أوديجارد هيلعب ضد مين الأسبوع ده؟",
        "reference":  "أوديجارد لاعب أرسنال ينتظر التصريح الطبي",
        "intent":     "player_info",
        "difficulty": "medium",
        "keywords":   ["أوديجارد", "أرسنال"],
    },
    {
        "question":   "مين سجل هدف بيراميدز ضد الجيش الملكي؟",
        "reference":  "هدف بيراميدز في مباراة الجيش الملكي",
        "intent":     "player_info",
        "difficulty": "hard",
        "keywords":   ["بيراميدز", "هدف"],
    },
    {
        "question":   "إيه إصابة تروسارد؟",
        "reference":  "تروسارد لاعب أرسنال ينتظر التصريح الطبي",
        "intent":     "player_info",
        "difficulty": "hard",
        "keywords":   ["تروسارد", "أرسنال"],
    },
    {
        "question":   "رايس هيجدد عقده مع أرسنال؟",
        "reference":  "أرسنال يبدأ مفاوضات تجديد عقد رايس وتيمبر",
        "intent":     "player_info",
        "difficulty": "medium",
        "keywords":   ["رايس", "أرسنال", "عقد"],
    },
    {
        "question":   "ماكتوميناي هيفضل في نابولي؟",
        "reference":  "نابولي لم يتفاوض مع ماكتوميناي لتجديد عقده",
        "intent":     "player_info",
        "difficulty": "medium",
        "keywords":   ["ماكتوميناي", "نابولي", "عقد"],
    },

    # ── Team News (10) ──────────────────────────────────────────────────────
    {
        "question":   "إيه تشكيل بيراميدز ضد الجيش الملكي؟",
        "reference":  "تشكيل بيراميدز يضم توفيق ظهير وناصر ماهر",
        "intent":     "team_news",
        "difficulty": "easy",
        "keywords":   ["بيراميدز", "تشكيل", "توفيق ظهير"],
    },
    {
        "question":   "تشكيل الجيش الملكي في مباراة بيراميدز؟",
        "reference":  "تشكيل الجيش الملكي يضم يوسف الفحلي",
        "intent":     "team_news",
        "difficulty": "easy",
        "keywords":   ["الجيش الملكي", "تشكيل", "يوسف الفحلي"],
    },
    {
        "question":   "إيه أخبار مران الأهلي النهارده؟",
        "reference":  "مران الأهلي يتضمن محاضرة فنية وتدريبات للحراس",
        "intent":     "team_news",
        "difficulty": "easy",
        "keywords":   ["الأهلي", "مران", "محاضرة"],
    },
    {
        "question":   "إيه بيعمل معتمد جمال مع الزمالك؟",
        "reference":  "معتمد جمال مدرب الزمالك قدم محاضرة فنية للاعبين",
        "intent":     "team_news",
        "difficulty": "medium",
        "keywords":   ["معتمد جمال", "الزمالك", "محاضرة"],
    },
    {
        "question":   "التحضيرات الأخيرة لأرسنال قبل مباراة إيفرتون؟",
        "reference":  "أرتيتا يؤكد جاهزية بن وايت لمباراة إيفرتون",
        "intent":     "team_news",
        "difficulty": "medium",
        "keywords":   ["أرسنال", "إيفرتون", "أرتيتا"],
    },
    {
        "question":   "فابريجاس قال إيه عن مانشستر سيتي؟",
        "reference":  "فابريجاس قال أن مانشستر سيتي لا يبني اللعب دائما من الخلف",
        "intent":     "team_news",
        "difficulty": "medium",
        "keywords":   ["فابريجاس", "مانشستر سيتي"],
    },
    {
        "question":   "كاريك قال إيه في مؤتمر الصحفي؟",
        "reference":  "مؤتمر كاريك يؤكد جاهزية ماونت",
        "intent":     "team_news",
        "difficulty": "hard",
        "keywords":   ["كاريك", "مؤتمر", "ماونت"],
    },
    {
        "question":   "إيه اللي بيحصل في مران الزمالك؟",
        "reference":  "تدريبات الزمالك قبل مباراة أوتوهو",
        "intent":     "team_news",
        "difficulty": "medium",
        "keywords":   ["الزمالك", "مران", "أوتوهو"],
    },
    {
        "question":   "الأهلي فاز على الزمالك في كرة اليد؟",
        "reference":  "الأهلي انتصر على الزمالك في دوري المحترفين لكرة اليد",
        "intent":     "team_news",
        "difficulty": "hard",
        "keywords":   ["الأهلي", "الزمالك", "كرة اليد"],
    },
    {
        "question":   "مدرب الجيش الملكي علق على إيه بعد مباراة بيراميدز؟",
        "reference":  "مدرب الجيش الملكي غير راضٍ عما حدث أمام بيراميدز",
        "intent":     "team_news",
        "difficulty": "easy",
        "keywords":   ["مدرب الجيش الملكي", "بيراميدز"],
    },

    # ── Transfer News (10) ──────────────────────────────────────────────────
    {
        "question":   "أرسنال هيجدد عقد رايس وتيمبر؟",
        "reference":  "أرسنال يبدأ مفاوضات لتجديد عقد رايس وتيمبر",
        "intent":     "transfer_news",
        "difficulty": "easy",
        "keywords":   ["أرسنال", "رايس", "تيمبر", "تجديد عقد"],
    },
    {
        "question":   "نابولي هيجدد مع ماكتوميناي؟",
        "reference":  "نابولي لم يتفاوض مع ماكتوميناي لتجديد عقده",
        "intent":     "transfer_news",
        "difficulty": "easy",
        "keywords":   ["نابولي", "ماكتوميناي", "عقد"],
    },
    {
        "question":   "غزل المحلة ضم مدرب جديد؟",
        "reference":  "غزل المحلة يعين سيد معوض مدرباً عاماً للفريق",
        "intent":     "transfer_news",
        "difficulty": "medium",
        "keywords":   ["غزل المحلة", "مدرب", "سيد معوض"],
    },
    {
        "question":   "في صفقات جديدة في الدوري المصري؟",
        "reference":  "صفقات الدوري المصري الأخيرة",
        "intent":     "transfer_news",
        "difficulty": "hard",
        "keywords":   ["الدوري المصري", "صفقة", "انتقال"],
    },
    {
        "question":   "مين جه جديد على أرسنال؟",
        "reference":  "صفقات أرسنال الجديدة",
        "intent":     "transfer_news",
        "difficulty": "hard",
        "keywords":   ["أرسنال", "انتقال"],
    },
    {
        "question":   "الزمالك هيجيب حد في الميركاتو؟",
        "reference":  "صفقات الزمالك في ميركاتو الشتاء",
        "intent":     "transfer_news",
        "difficulty": "hard",
        "keywords":   ["الزمالك", "ميركاتو"],
    },
    {
        "question":   "أهلي جدة ضم لاعبين جدد؟",
        "reference":  "صفقات أهلي جدة الأخيرة",
        "intent":     "transfer_news",
        "difficulty": "hard",
        "keywords":   ["أهلي جدة"],
    },
    {
        "question":   "هل بيراميدز هيضم لاعب جديد؟",
        "reference":  "صفقات بيراميدز في الميركاتو",
        "intent":     "transfer_news",
        "difficulty": "hard",
        "keywords":   ["بيراميدز", "ضم"],
    },
    {
        "question":   "صلاح هيجدد مع ليفربول؟",
        "reference":  "مفاوضات تجديد عقد صلاح مع ليفربول",
        "intent":     "transfer_news",
        "difficulty": "medium",
        "keywords":   ["صلاح", "ليفربول", "عقد"],
    },
    {
        "question":   "مين المدرب الجديد للمنتخب المصري؟",
        "reference":  "المدرب الجديد للمنتخب المصري",
        "intent":     "transfer_news",
        "difficulty": "hard",
        "keywords":   ["المنتخب المصري", "مدرب"],
    },

    # ── General Football (10) ────────────────────────────────────────────────
    {
        "question":   "ترتيب الدوري المصري الحالي؟",
        "reference":  "ترتيب فرق الدوري المصري الممتاز",
        "intent":     "general_football",
        "difficulty": "medium",
        "keywords":   ["الدوري المصري", "ترتيب"],
    },
    {
        "question":   "ترتيب المجموعة في دوري أبطال إفريقيا؟",
        "reference":  "ترتيب مجموعات دوري أبطال إفريقيا",
        "intent":     "general_football",
        "difficulty": "hard",
        "keywords":   ["أبطال إفريقيا", "مجموعة"],
    },
    {
        "question":   "موعد مباريات الأسبوع القادم؟",
        "reference":  "مباريات الجولة القادمة",
        "intent":     "general_football",
        "difficulty": "medium",
        "keywords":   ["مباريات", "موعد"],
    },
    {
        "question":   "أخبار كرة القدم المصرية اليوم؟",
        "reference":  "آخر أخبار الكرة المصرية",
        "intent":     "general_football",
        "difficulty": "easy",
        "keywords":   ["الكرة المصرية", "أخبار"],
    },
    {
        "question":   "إيه اللي بيحصل في الدوري الإنجليزي الأسبوع ده؟",
        "reference":  "أخبار الدوري الإنجليزي الممتاز",
        "intent":     "general_football",
        "difficulty": "medium",
        "keywords":   ["الدوري الإنجليزي"],
    },
    {
        "question":   "إيه أخبار أبطال أوروبا؟",
        "reference":  "أخبار دوري أبطال أوروبا",
        "intent":     "general_football",
        "difficulty": "medium",
        "keywords":   ["أبطال أوروبا"],
    },
    {
        "question":   "الكأس الأفريقية مين المتأهل؟",
        "reference":  "المتأهلون لكأس الأمم الأفريقية",
        "intent":     "general_football",
        "difficulty": "hard",
        "keywords":   ["الكأس الأفريقية", "تأهل"],
    },
    {
        "question":   "كاريك مدرب منتخب إيه؟",
        "reference":  "كاريك المدير الفني",
        "intent":     "general_football",
        "difficulty": "medium",
        "keywords":   ["كاريك", "مدرب"],
    },
    {
        "question":   "يورتشيتش مدرب بيراميدز قال إيه بعد المباراة؟",
        "reference":  "يورتشيتش مدرب بيراميدز علق على المباراة",
        "intent":     "general_football",
        "difficulty": "easy",
        "keywords":   ["يورتشيتش", "بيراميدز"],
    },
    {
        "question":   "روسينيور قال إيه عن الأخطاء؟",
        "reference":  "روسينيور قال أن الأخطاء جزء من كرة القدم",
        "intent":     "general_football",
        "difficulty": "medium",
        "keywords":   ["روسينيور", "أخطاء"],
    },
]


# ════════════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════════════

def rouge1_f1(prediction: str, reference: str) -> float:
    """ROUGE-1 F1 score (no deps, Arabic-aware tokenization)."""
    tok = lambda t: set(re.findall(r'[\u0600-\u06FF\w]+', t.lower()))
    pred_tok = tok(prediction)
    ref_tok  = tok(reference)
    if not ref_tok or not pred_tok:
        return 0.0
    overlap   = pred_tok & ref_tok
    precision = len(overlap) / len(pred_tok)
    recall    = len(overlap) / len(ref_tok)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def keyword_hit(retrieved_texts: list[str], keywords: list[str]) -> float:
    """
    Keyword-based hit rate: fraction of required keywords found in retrieved chunks.
    More reliable than ROUGE for Arabic since we know ground-truth keywords.
    """
    combined = ' '.join(retrieved_texts).lower()
    if not keywords:
        return 1.0
    hits = sum(1 for kw in keywords if kw.lower() in combined)
    return hits / len(keywords)


def mrr_score(retrieved_texts: list[str], keywords: list[str]) -> float:
    """
    Mean Reciprocal Rank: score = 1/rank of first chunk that contains any keyword.
    """
    for rank, text in enumerate(retrieved_texts, 1):
        if any(kw.lower() in text.lower() for kw in keywords):
            return 1.0 / rank
    return 0.0


def intent_accuracy(predicted: str, expected: str) -> float:
    return 1.0 if predicted == expected else 0.0


# ════════════════════════════════════════════════════════════════════════════
# Experiment Result
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentResult:
    name:           str
    rouge_scores:   list[float] = field(default_factory=list)
    keyword_hits:   list[float] = field(default_factory=list)
    mrr_scores:     list[float] = field(default_factory=list)
    latencies_ms:   list[float] = field(default_factory=list)
    intents:        list[str]   = field(default_factory=list)

    # Per-intent breakdown: {intent: [scores]}
    per_intent_rouge:    dict = field(default_factory=lambda: defaultdict(list))
    per_intent_keywords: dict = field(default_factory=lambda: defaultdict(list))

    @property
    def avg_rouge(self) -> float:
        return _avg(self.rouge_scores)

    @property
    def avg_keyword_hit(self) -> float:
        return _avg(self.keyword_hits)

    @property
    def avg_mrr(self) -> float:
        return _avg(self.mrr_scores)

    @property
    def avg_latency(self) -> float:
        return _avg(self.latencies_ms)

    def per_intent_summary(self) -> dict:
        result = {}
        for intent in set(self.intents):
            r = self.per_intent_rouge.get(intent, [])
            k = self.per_intent_keywords.get(intent, [])
            result[intent] = {
                "rouge1":      round(_avg(r), 3),
                "keyword_hit": round(_avg(k), 3),
                "n_cases":     len(r),
            }
        return result


def _avg(lst: list) -> float:
    return sum(lst) / len(lst) if lst else 0.0


# ════════════════════════════════════════════════════════════════════════════
# Ablation: Retrieval-only evaluation
# ════════════════════════════════════════════════════════════════════════════

def run_retrieval_experiment(
    test_cases: list[dict],
    use_bm25:   bool,
    use_dense:  bool,
    name:       str,
) -> ExperimentResult:
    """
    Evaluate retrieval quality for a given configuration.
    Does NOT call Groq — only tests retriever.
    """
    from retrieval.hybrid_retriever import FilGoalRetriever

    result    = ExperimentResult(name=name)
    retriever = FilGoalRetriever()
    retriever.load()

    # Disable components for ablation
    if not use_bm25:
        retriever.bm25 = None
    if not use_dense:
        retriever.index = None
        retriever.model = None

    for case in test_cases:
        q        = case["question"]
        ref      = case["reference"]
        keywords = case.get("keywords", [])
        intent   = case.get("intent", "general_football")

        t0     = time.time()
        chunks = retriever.retrieve(q, top_k=5)
        ms     = (time.time() - t0) * 1000

        texts = [c.get("text", "") for c in chunks]
        r1    = rouge1_f1(" ".join(texts), ref)
        kh    = keyword_hit(texts, keywords)
        mrr   = mrr_score(texts, keywords)

        result.rouge_scores.append(r1)
        result.keyword_hits.append(kh)
        result.mrr_scores.append(mrr)
        result.latencies_ms.append(ms)
        result.intents.append(intent)
        result.per_intent_rouge[intent].append(r1)
        result.per_intent_keywords[intent].append(kh)

    return result


# ════════════════════════════════════════════════════════════════════════════
# End-to-End RAG Evaluation (optional, requires Groq)
# ════════════════════════════════════════════════════════════════════════════

def run_rag_evaluation(test_cases: list[dict]) -> dict:
    """
    Full RAG evaluation: retriever + LLM answer quality.
    Also measures intent detection accuracy.
    """
    from qa_engine.rag_pipeline import FilGoalRAG, detect_intent

    bot = FilGoalRAG()
    bot.load()

    intent_correct = 0
    rouge_scores   = []
    keyword_hits   = []
    latencies      = []
    per_intent     = defaultdict(lambda: {"rouge": [], "keywords": [], "intent_acc": []})

    for i, case in enumerate(test_cases):
        q        = case["question"]
        ref      = case["reference"]
        keywords = case.get("keywords", [])
        expected = case.get("intent", "general_football")

        log.info(f"  [{i+1}/{len(test_cases)}] {q[:50]}...")

        t0     = time.time()
        result = bot.answer(q)
        ms     = (time.time() - t0) * 1000

        answer          = result.get("answer", "")
        predicted_intent = result.get("intent", "")

        # Metrics
        r1  = rouge1_f1(answer, ref)
        kh  = keyword_hit([answer] + [s.get("title","") for s in result.get("sources",[])], keywords)
        ia  = intent_accuracy(predicted_intent, expected)

        rouge_scores.append(r1)
        keyword_hits.append(kh)
        latencies.append(ms)
        intent_correct += ia

        per_intent[expected]["rouge"].append(r1)
        per_intent[expected]["keywords"].append(kh)
        per_intent[expected]["intent_acc"].append(ia)

    n = len(test_cases)
    summary = {
        "n_cases":          n,
        "avg_rouge1":       round(_avg(rouge_scores), 3),
        "avg_keyword_hit":  round(_avg(keyword_hits), 3),
        "intent_accuracy":  round(intent_correct / n, 3),
        "avg_latency_ms":   round(_avg(latencies), 1),
        "per_intent": {
            intent: {
                "avg_rouge1":      round(_avg(v["rouge"]), 3),
                "avg_keyword_hit": round(_avg(v["keywords"]), 3),
                "intent_accuracy": round(_avg(v["intent_acc"]), 3),
                "n_cases":         len(v["rouge"]),
            }
            for intent, v in per_intent.items()
        }
    }
    return summary


# ════════════════════════════════════════════════════════════════════════════
# Print helpers
# ════════════════════════════════════════════════════════════════════════════

def print_ablation_table(results: list[ExperimentResult]):
    print(f"\n{'='*70}")
    print("ABLATION STUDY — Retrieval Quality")
    print(f"{'='*70}")
    header = f"{'Experiment':<32} {'ROUGE-1':>8} {'Kw-Hit':>8} {'MRR':>8} {'Latency':>10}"
    print(header)
    print("-" * 70)
    for r in results:
        row = (
            f"{r.name:<32} "
            f"{r.avg_rouge:>8.3f} "
            f"{r.avg_keyword_hit:>8.3f} "
            f"{r.avg_mrr:>8.3f} "
            f"{r.avg_latency:>8.0f}ms"
        )
        print(row)
    print("-" * 70)

    # Highlight winner
    best = max(results, key=lambda x: x.avg_keyword_hit)
    print(f"\n🏆 Best retriever: {best.name}  (keyword hit = {best.avg_keyword_hit:.3f})")


def print_intent_breakdown(result: ExperimentResult):
    print(f"\n{'='*70}")
    print(f"PER-INTENT BREAKDOWN — {result.name}")
    print(f"{'='*70}")
    header = f"{'Intent':<22} {'ROUGE-1':>8} {'Kw-Hit':>8} {'N':>5}"
    print(header)
    print("-" * 50)
    for intent, metrics in sorted(result.per_intent_summary().items()):
        row = (
            f"{intent:<22} "
            f"{metrics['rouge1']:>8.3f} "
            f"{metrics['keyword_hit']:>8.3f} "
            f"{metrics['n_cases']:>5}"
        )
        print(row)


def print_rag_results(rag: dict):
    print(f"\n{'='*70}")
    print("END-TO-END RAG EVALUATION")
    print(f"{'='*70}")
    print(f"  Cases:            {rag['n_cases']}")
    print(f"  ROUGE-1:          {rag['avg_rouge1']:.3f}")
    print(f"  Keyword Hit Rate: {rag['avg_keyword_hit']:.3f}")
    print(f"  Intent Accuracy:  {rag['intent_accuracy']:.3f}")
    print(f"  Avg Latency:      {rag['avg_latency_ms']:.0f}ms")
    print(f"\n  Per-Intent Breakdown:")
    print(f"  {'Intent':<22} {'ROUGE-1':>8} {'Kw-Hit':>8} {'Intent-Acc':>12} {'N':>5}")
    print("  " + "-" * 60)
    for intent, m in sorted(rag["per_intent"].items()):
        print(
            f"  {intent:<22} "
            f"{m['avg_rouge1']:>8.3f} "
            f"{m['avg_keyword_hit']:>8.3f} "
            f"{m['intent_accuracy']:>12.3f} "
            f"{m['n_cases']:>5}"
        )


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def run_evaluation(run_rag: bool = False, save_report: bool = False):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Load or create test set
    if TEST_SET_PATH.exists():
        test_cases = json.loads(TEST_SET_PATH.read_text(encoding='utf-8'))
        log.info(f"✅ Loaded {len(test_cases)} test cases from {TEST_SET_PATH}")
    else:
        test_cases = DEFAULT_TEST_SET
        TEST_SET_PATH.write_text(
            json.dumps(test_cases, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        log.info(f"📝 Created default test set: {TEST_SET_PATH}")

    # ── Ablation experiments ──────────────────────────────────────────────
    ablation_configs = [
        ("BM25 only (baseline)",    True,  False),
        ("Dense only (FAISS)",      False, True),
        ("Hybrid BM25+FAISS (RRF)", True,  True),
    ]

    ablation_results = []
    for name, use_bm25, use_dense in ablation_configs:
        log.info(f"\n▶ Running: {name}")
        r = run_retrieval_experiment(test_cases, use_bm25, use_dense, name)
        ablation_results.append(r)
        log.info(
            f"   ROUGE-1={r.avg_rouge:.3f}  "
            f"Kw-Hit={r.avg_keyword_hit:.3f}  "
            f"MRR={r.avg_mrr:.3f}  "
            f"Latency={r.avg_latency:.0f}ms"
        )

    print_ablation_table(ablation_results)

    # Best retriever intent breakdown
    best = max(ablation_results, key=lambda x: x.avg_keyword_hit)
    print_intent_breakdown(best)

    # ── End-to-end RAG ────────────────────────────────────────────────────
    rag_summary = None
    if run_rag:
        if not os.getenv("GROQ_API_KEY"):
            log.warning("GROQ_API_KEY not set — skipping end-to-end RAG eval")
        else:
            log.info("\n▶ Running end-to-end RAG evaluation...")
            rag_summary = run_rag_evaluation(test_cases)
            print_rag_results(rag_summary)

    # ── Save report ───────────────────────────────────────────────────────
    if save_report:
        report = {
            "ablation": [
                {
                    "name":          r.name,
                    "avg_rouge1":    round(r.avg_rouge, 3),
                    "avg_kw_hit":    round(r.avg_keyword_hit, 3),
                    "avg_mrr":       round(r.avg_mrr, 3),
                    "avg_latency_ms":round(r.avg_latency, 1),
                    "per_intent":    r.per_intent_summary(),
                }
                for r in ablation_results
            ],
            "rag": rag_summary,
        }
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        log.info(f"\n💾 Report saved: {REPORT_PATH}")

    log.info("\n✅ Evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FilGoalBot Evaluation")
    parser.add_argument("--rag",         action="store_true", help="Run full RAG evaluation (needs GROQ_API_KEY)")
    parser.add_argument("--save-report", action="store_true", help="Save JSON report to evaluation/report.json")
    args = parser.parse_args()

    run_evaluation(run_rag=args.rag, save_report=args.save_report)
