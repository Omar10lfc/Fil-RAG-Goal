"""
FilGoalBot RAG Pipeline
========================
Retrieve hybrid chunks → build context → call Groq → answer with citations.

Intent routing lives in qa_engine/intent.py.
Prompts live in qa_engine/prompts.py.
LLM response cache lives in qa_engine/cache.py.
"""

import logging
import os

from dotenv import load_dotenv
from groq import Groq

from qa_engine import cache, prompts
from qa_engine.intent import EXTRACTIVE_INTENTS, detect_intent
from retrieval.hybrid_retriever import FilGoalRetriever

load_dotenv()
log = logging.getLogger("rag")

# ── Models ────────────────────────────────────────────────────────────────────
# Extractive intents (lineup, match_result) get the cheap 8B model: the answer
# is a direct fact in the retrieved chunk, the LLM only formats it.
# Everything else gets the 70B for reasoning across multiple chunks.
GROQ_MODEL_BIG    = "llama-3.3-70b-versatile"
GROQ_MODEL_SMALL  = "llama-3.1-8b-instant"

MAX_TOKENS  = 600
TEMPERATURE = 0.2
TOP_K       = 6

# Refuse to send a request whose prompt + max_tokens would exceed this. Groq's
# free tier is 100k tokens/day; bailing locally beats the 429 retry storm.
MAX_PROMPT_TOKENS = 6000

FILTER_MAP = {
    "lineup":        {"filter_type": "lineup"},
    "match_result":  {"filter_type": "match_result"},
    "transfer_news": {"filter_type": "transfer"},
}


def _model_for(intent: str) -> str:
    return GROQ_MODEL_SMALL if intent in EXTRACTIVE_INTENTS else GROQ_MODEL_BIG


def _build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    """Build LLM context string + sources list from retrieved chunks.

    Uses body_clean (readable Arabic). The text field is the normalised
    embedding string — alef variants stripped, ة→ه — and would garble answers.
    """
    context_parts: list[str] = []
    sources: list[dict] = []

    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("title", "")
        date  = chunk.get("pub_date", "")[:10]

        body = chunk.get("body_clean") or chunk.get("text", "")
        if not body:
            raw = chunk.get("text", "")
            parts = raw.split("\n\n", 1)
            body = parts[1].strip() if len(parts) > 1 else raw

        # Lineups are list-shaped and need more room; news is dense.
        body = body[:1500] if chunk.get("article_type") == "lineup" else body[:800]

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
            "chunk_id":     chunk.get("chunk_id", ""),
        })

    return "\n\n---\n\n".join(context_parts), sources


class FilGoalRAG:
    def __init__(self, retriever: FilGoalRetriever | None = None):
        # When a pre-loaded retriever is passed in (e.g. from the eval suite),
        # skip retriever.load() to avoid a second FAISS+BM25+ST cold start.
        self._retriever_provided = retriever is not None
        self.retriever = retriever or FilGoalRetriever()
        self.groq = None  # initialised in load() after .env is confirmed

    def load(self):
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set.\n"
                "Add it to your .env file:\n"
                "  GROQ_API_KEY=gsk_..."
            )
        self.groq = Groq(api_key=api_key)
        log.info("✅ Groq client ready")

        if not self._retriever_provided:
            self.retriever.load()
        log.info("FilGoalRAG ready")

    def answer(
        self,
        query: str,
        filter_type: str | None = None,
        filter_league: str | None = None,
        filter_team: str | None = None,
    ) -> dict:
        intent  = detect_intent(query)
        filters = dict(FILTER_MAP.get(intent, {}))
        if filter_type   is not None: filters["filter_type"]   = filter_type
        if filter_league is not None: filters["filter_league"] = filter_league
        if filter_team   is not None: filters["filter_team"]   = filter_team
        log.info(f"Intent: {intent} | Filters: {filters} | Query: {query}")

        chunks = self.retriever.retrieve(query, top_k=TOP_K, **filters)
        if not chunks:
            return {
                "answer":  "لا تتوفر لديّ معلومات كافية للإجابة على هذا السؤال.",
                "intent":  intent,
                "sources": [],
                "model":   None,
                "cached":  False,
            }

        context, sources = _build_context(chunks)
        chunk_ids = [c.get("chunk_id", "") for c in chunks]
        model = _model_for(intent)

        # ── Cache lookup ──────────────────────────────────────────────────────
        cached = cache.get(model, intent, chunk_ids, query)
        if cached is not None:
            log.info(f"  ↩ cache hit ({model})")
            return {
                "answer":  cached,
                "intent":  intent,
                "sources": sources[:3],
                "model":   model,
                "cached":  True,
            }

        # ── Token budget guard ────────────────────────────────────────────────
        system_prompt = prompts.INTENT_PROMPTS[intent]
        user_prompt   = f"السياق:\n\n{context}\n\n{'─'*40}\n\nالسؤال: {query}"
        prompt_tokens = cache.estimate_tokens(system_prompt) + cache.estimate_tokens(user_prompt)
        if prompt_tokens + MAX_TOKENS > MAX_PROMPT_TOKENS:
            log.warning(f"  prompt too large ({prompt_tokens} est. tokens) — truncating context")
            # Halve the context — naive but predictable
            context = context[: len(context) // 2]
            user_prompt = f"السياق:\n\n{context}\n\n{'─'*40}\n\nالسؤال: {query}"

        # ── Call Groq ─────────────────────────────────────────────────────────
        try:
            response = self.groq.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            answer_text = response.choices[0].message.content.strip()
            cache.put(model, intent, chunk_ids, query, answer_text)
        except Exception as e:
            log.error(f"Groq error: {e}")
            answer_text = "حدث خطأ أثناء توليد الإجابة، يرجى المحاولة مرة أخرى."

        return {
            "answer":  answer_text,
            "intent":  intent,
            "sources": sources[:3],
            "model":   model,
            "cached":  False,
        }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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
        print(f"\n{'='*60}\n {q}")
        try:
            result = bot.answer(q)
            print(f"Intent : {result['intent']}  (model={result['model']}, cached={result['cached']})")
            print(f"Answer : {result['answer']}")
            print(f"Sources: {[s['title'][:45] for s in result['sources']]}")
        except Exception as e:
            log.error(f"❌ Question failed: {e}")
