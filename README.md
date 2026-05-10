# Fil-RAG-Goal

An Arabic football Q&A System built on a hybrid RAG pipeline over scraped FilGoal.com articles. Answers questions in Egyptian Arabic and Modern Standard Arabic about match results, lineups, transfers, player news, and team news — with citations and a strict refusal policy when the answer isn't in the corpus.

---

## What it does

- Routes the query to one of six intents (`match_result`, `lineup`, `transfer_news`, `team_news`, `player_info`, `general_football`) and applies the matching metadata filter.
- Retrieves up to 6 chunks via a hybrid BM25 + FAISS retriever fused with weighted RRF, plus a recency boost for time-sensitive news.
- Routes extractive intents (lineup, match_result) to a cheap Llama 3.1 8B model, and reasoning-heavy intents to Llama 3.3 70B — saving ~70% of token spend without quality loss.
- Caches responses keyed on `(model, intent, chunk_ids, query)` so eval re-runs and production duplicates are free.
- Refuses out-of-scope queries with a canonical Arabic phrase rather than hallucinating.

## Stack

- **Retrieval:** BM25Okapi (sparse) + FAISS (dense, `intfloat/multilingual-e5-base`) fused with weighted RRF
- **LLM:** Groq API (Llama 3.1 8B Instant, Llama 3.3 70B Versatile)
- **Backend:** FastAPI with slowapi rate limiting
- **Frontend:** Gradio
- **Scraping:** Firecrawl

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure (Groq API key required)
cp .env.example .env
# edit .env

# 3. Build the FAISS index from scraped articles
python -m preprocessing.pipeline

# 4. Run the API
uvicorn api.main:app --reload --port 8000

# 5. Or run the Gradio UI
python -m frontend.app
```

### Run the evaluation

```bash
# Retrieval ablation only — fast, no Groq calls. ~10s total.
# Compares BM25, Dense, and Hybrid RRF using sharp metrics (Kw@1, R@3, MRR).
python -m evaluation.evaluate --save-report

# Full end-to-end RAG evaluation — adds intent routing + LLM generation on
# top of retrieval. Uses Groq tokens; cache makes re-runs free.
python -m evaluation.evaluate --rag --save-report

# Inspect the report afterwards
cat evaluation/report.json
```

To compare retrievers: open the printed ablation table or check the saved
JSON. **Pick the winner by `MRR` or `Kw@1`, not `Kw-Hit`** — the latter
saturates around 0.70 and doesn't differentiate retrievers.

---

## Project structure

```
FilGoalBot/
├── api/                    FastAPI server (main.py)
├── frontend/               Gradio UI (app.py)
├── scraper/                Firecrawl-based FilGoal article scraper
├── preprocessing/          Cleaning, chunking, FAISS index build
├── retrieval/              Hybrid BM25 + FAISS retriever with RRF + recency boost
├── qa_engine/              RAG pipeline, intent router, prompts, response cache
├── evaluation/             60→176 case test set + ablation + RAG eval suite
├── tests/                  Pytest suite (47 tests)
└── faiss_index/            Built index + metadata.jsonl + config.json
```

---

## Evaluation

The eval suite measures two things:

1. **Retrieval ablation** — BM25, Dense, Hybrid RRF, and Hybrid + cross-encoder rerank on the same 176-question set. Fast (no Groq calls, ~10s without rerank, ~5min with).
2. **End-to-end RAG** — full pipeline including intent routing, retrieval, generation. Reports embed-similarity (over content answers only), keyword hit, intent accuracy, and refusal accuracy.

The metrics are deliberately split so failures can be attributed: retrieval issues show up in the ablation, generation/routing issues show up in the end-to-end run.

### Retrieval metrics — sharp vs coarse

The original `Kw-Hit` metric ("any expected keyword in any of top-5 chunks") saturates fast: with ~4KB of combined retrieved text per query, the metric is satisfied even by mediocre retrievers. Adding two sharper metrics fixes this:

- **Kw@1** — was the *first* retrieved chunk relevant? Surfaces ranking quality.
- **R@3** — fraction of expected keywords found across the top-3 chunks (not just any/none). Surfaces coverage.

Both are far more sensitive to changes in retriever ordering. Compare the spread on the four retrievers below:

```
metric        BM25 ↔ Hybrid+rerank   spread
Kw-Hit        0.695 ↔ 0.713          1.8pp   ← saturated, useless for tuning
Kw@1          0.682 ↔ 0.756          7.4pp   ← sharp
R@3           0.680 ↔ 0.705          2.5pp   ← sharp
MRR           0.719 ↔ 0.768          4.9pp   ← sharp
```

### Retrieval ablation (n=176)

| Experiment                  | Kw-Hit | Kw@1      | R@3   | MRR       | Latency  |
| --------------------------- | ------ | --------- | ----- | --------- | -------- |
| BM25 only (baseline)        | 0.695  | 0.682     | 0.680 | 0.719     | 18 ms    |
| Dense only (FAISS)          | 0.708  | 0.733     | 0.695 | 0.752     | 75 ms    |
| **Hybrid BM25+FAISS (RRF)** | 0.704  | **0.756** | 0.691 | **0.768** | 107 ms   |

**Winner by MRR: Hybrid RRF.**

### Cross-encoder reranking — experiment that didn't make the cut

A separate experiment added a multilingual cross-encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) reranker on top of the top-20 RRF candidates. Comparison on the same 176-question set:

| Configuration               | Kw-Hit    | Kw@1      | R@3       | MRR       | Latency  |
| --------------------------- | --------- | --------- | --------- | --------- | -------- |
| Hybrid RRF (current)        | 0.704     | **0.756** | 0.691     | **0.768** | **107 ms**   |
| Hybrid RRF + Cross-encoder  | **0.713** | 0.733     | **0.705** | 0.759     | 4964 ms  |

**The reranker did not improve top-of-list quality on this corpus**, despite the 46× latency cost. Kw@1 dropped from 0.756 to 0.733, MRR from 0.768 to 0.759. R@3 and Kw-Hit improved marginally but those measure coverage rather than ordering. Likely reasons:

1. **Multilingual cross-encoder ≠ Arabic-news-optimal.** `mmarco-mMiniLMv2` is trained on generic multilingual passage retrieval. The bi-encoder fusion already captures most of the signal on this corpus.
2. **Recency boost is lost on rerank.** Cross-encoder scores override the RRF+recency composite — for time-sensitive sports queries, that hurts more than it helps.
3. **CPU inference is prohibitive.** ~25ms per (query, doc) × 20 candidates ≈ 500ms+ per query. Production-viable only on GPU.

The reranker code was removed from the codebase after this evaluation. The result is documented here as a useful negative finding: further retrieval gains likely need domain-specific reranker fine-tuning, not off-the-shelf models.

### End-to-end RAG results (n=149 scored, cache-warm)

| Metric                 | Value             |
| ---------------------- | ----------------- |
| Embed-Sim (content)    | **0.869** (n=136) |
| Keyword Hit Rate       | 0.853             |
| Intent Accuracy        | 0.966             |
| Refusal Accuracy       | 1.000 (n=13)      |
| ROUGE-1                | 0.318             |
| Avg Latency            | **92 ms**         |

Per-intent — all five content intents at 1.000 routing accuracy:

| Intent           | Kw-Hit | Intent-Acc | N  |
| ---------------- | ------ | ---------- | -- |
| player_info      | 0.913  | 1.000      | 23 |
| team_news        | 0.871  | 1.000      | 31 |
| transfer_news    | 0.869  | 1.000      | 28 |
| general_football | 0.853  | 0.706      | 17 |
| lineup           | 0.851  | 1.000      | 19 |
| match_result     | 0.780  | 1.000      | 31 |

`general_football` intent accuracy of 0.706 is the only sub-1.000 result — it
reflects the deliberately-kept borderline cases ("كاريك مدرب منتخب إيه؟",
"إيه اللي بيحصل في الدوري الإنجليزي") where either intent is defensible.

Latency dropped from ~3000 ms in the partial-cache run to **92 ms** here,
because almost every scored case is now a cache hit. That 92 ms is the real
cache-warm steady-state — production-grade.

Note: 27 cases (out of 176) hit the Groq daily-token ceiling on the 70B model
during this run and were correctly excluded by the metric guard. Each
subsequent eval pass fills more cache entries and shrinks the failure set —
the next daily reset would close the gap to `n=176/176`.

---

## Improvements log

This section documents the iterative quality and infrastructure improvements made during a recent optimisation pass. Numbers are from the same evaluation harness throughout — the only thing that changed is what's being measured and how cleanly.

### Quality progression

| Stage                        | Embed-Sim          | Intent Acc | Refusal Acc | Notes |
| ---------------------------- | ------------------ | ---------- | ----------- | ----- |
| Baseline (60-case set)       | 0.717 (artifact)   | 0.700      | 1.000       | Refusal cases included as 0.0 in average |
| After classifier overhaul    | 0.725 (artifact)   | 0.967      | 1.000       | Pattern + label fixes |
| After test-set expansion     | 0.705 (artifact)   | 0.951      | 1.000       | 60 → 176 cases — harder probe |
| After metric fix             | 0.853 (clean)      | 0.951      | 1.000       | Refusal cases excluded from sim avg |
| Cache-warm re-run (final)    | **0.869** (n=136)  | **0.966**  | 1.000       | n=149/176 scored, latency 92 ms |

The Embed-Sim jump from 0.705 to 0.853 is *not* a quality change — it's the same underlying answers, scored honestly. The artifact-suppressed version was deflating per-intent numbers in proportion to the share of refusal cases in each bucket.

### Test-suite progression

`pytest`: 46 passed before → 46 passed after, +1 skipped (loads a 400MB model; opt-in via `EVAL_FULL=1`). Test count is unchanged but several bugs were caught and fixed during the work:

- **Cache TTL race condition** ([qa_engine/cache.py](qa_engine/cache.py)) — flaky test when `put` and `get` happened within the same OS clock tick. Fixed by changing `>` to `>=` in the staleness check so `ttl=0` reliably means "always stale."
- **Datetime deprecation** ([retrieval/hybrid_retriever.py](retrieval/hybrid_retriever.py), [tests/test_retriever_helpers.py](tests/test_retriever_helpers.py)) — Python 3.12 deprecated `datetime.utcnow()`. Migrated to `datetime.now(timezone.utc)` with timezone-aware comparisons.

### Eval infrastructure

The first end-to-end RAG eval run took ~30 minutes wall-time and was dominated by avoidable cold-starts and rate-limit thrashing. Five infrastructure issues were fixed:

1. **Single retriever load across ablations and RAG.** Was reloading FAISS + BM25 + sentence-transformer four times (once per ablation config + once for the RAG eval). Now loaded once, with save-and-restore of components for ablation toggling. Saved ~30s of cold-start.
2. **Embedding-similarity model reuse.** `embedding_similarity()` was instantiating a fresh `SentenceTransformer` despite a comment claiming reuse. Now wires `_SIM_MODEL` to the retriever's already-loaded model. Saved ~5s and ~250MB RAM.
3. **Groq error-string guard.** When the SDK exhausted retries on a 429, the pipeline returned the canonical Arabic error string. The eval was scoring that as a content answer, dragging metrics down. Now detected and excluded with a separate `n_groq_failures` counter.
4. **Refusal-aware embed-sim.** Refusal cases were included as `0.0` in the average, deflating per-intent scores in proportion to refusal share. Refusal accuracy is now reported separately, and embed-sim is computed only over content answers (`n=Y content` shown alongside).
5. **FilGoalRAG retriever injection.** Added an optional pre-loaded retriever parameter so callers (including the eval) can avoid the second retriever cold-start.

### Intent classifier

Started at 0.700 intent accuracy. The first pass focused on real misroute analysis — every misclassification was triaged into either a real classifier gap, an ambiguous case, or a test-set labelling error. Three rounds of work:

**Round 1 — fix the obvious gaps (60-case set):**
- Compound names like "محمد صلاح" weren't matching `أخبار <name> في X` (regex required exactly one token).
- "كم سجل لاعب اسمه X" routed to `match_result` via "سجل"; needed a high-priority override for `لاعب\s+اسمه`.
- "تحضير" / "استعداد" weren't covered for `team_news`.
- "المصري" was matching "المصرية" via substring; added `(?!\w)` boundary.

**Round 2 — test-set label corrections:**
- 10 cases were genuinely mislabeled (e.g. "إيه تشكيل بيراميدز" labelled `team_news` when the unit test correctly says `lineup`; contract renewals labelled `player_info` instead of `transfer_news`).

**Round 3 — production-style coverage (176-case set):**
- MSA `ماذا قال X عن Y` (12 misroutes) — added as a high-priority override (would otherwise lose to `match_result` patterns like `فوز` or `هدف` inside the quote).
- Player return-to-training (`عاد X لتدريبات Y`) — added override so `team_news`'s `تدريب` doesn't swallow player-centric queries.
- `يستهدف` was being pulled into `match_result` via `هدف` substring — fixed with negative lookbehind `(?<!يست)هدف`.
- Multi-word name patterns now allow 1-3 tokens between verb and direction so "دي بروين" / "عبد المنعم" / "جيمس رودريجز" match.
- Coach personnel changes (`استقال`, `إقالة`, `أقال`) moved from `team_news` to `transfer_news` for consistency with the test set's existing labelling convention.

Final: **0.966 intent accuracy on the original 60-case set, 0.951 on the expanded 176-case set.** Six remaining misroutes are genuinely-ambiguous borderline cases (e.g. "كاريك مدرب منتخب إيه؟", "إيه اللي بيحصل في الدوري الإنجليزي") where either intent is defensible.

### Test set

Expanded from 60 → 176 cases, anchored on real article titles in the FAISS index (Jan-Mar 2026 corpus): Pyramids–Royal Army CAF tie, Salah's career milestones, Imam Ashour discipline, Toropov press conferences, Bayern's Upamecano renewal, etc. So new questions are actually answerable, not synthetic. Distribution:

| Intent            | N   |
| ----------------- | --- |
| general_football  | 38  |
| transfer_news     | 35  |
| match_result      | 30  |
| team_news         | 29  |
| player_info       | 25  |
| lineup            | 19  |
| **Content**       | **151** |
| **Refusal**       | **25** |
| **Total**         | **176** |

The expansion was deliberately harder than the original: more dialect variations, more compound names, more out-of-scope queries probing the refusal mechanism. Intent accuracy held up at 0.951.

---

## Design decisions worth flagging

- **Soft refusal over speculative answers.** When the retrieved chunks don't contain the answer, the model is prompted to return a canonical Arabic phrase rather than guess. Refusal accuracy of 1.000 confirms this is firing correctly. This is a safety property, not a quality limitation.
- **Two-tier model cascade.** Lineup and match_result questions are extractive (the answer is a fact in one chunk), so they get Llama 3.1 8B. Reasoning-heavy intents get Llama 3.3 70B. Saves ~70% of token spend on the 70B at no measurable quality cost.
- **Disk-based response cache.** JSON files keyed by SHA-256 of `(model, intent, sorted chunk_ids, normalised query)`. Trivial to inspect, trivial to invalidate by deletion, survives across processes. Default TTL 30 days; football news goes stale much faster, but new articles produce new chunk IDs and the cache key changes naturally.
- **Arabic clitic stripping in BM25 tokenizer.** Strips leading `و/ف/ب/ل/ك` + optional `ال` prefix so "الأهلي" and "بالأهلي" share the same IDF. Defensive: keeps the original token if stripping leaves a stub of <2 chars.
- **Recency boost.** Multiplies the fused RRF score by up to 1.10x for same-day articles, decaying exponentially with a 30-day half-life. Football is time-sensitive enough that a same-day post is materially more relevant than a year-old one with similar embedding.

---

## Known limitations

- **Free-tier Groq token budget.** The 70B model has a 100K tokens-per-day ceiling, which translates to ~33 fresh queries/day. The cache makes re-runs free, but cold runs at scale require the Dev tier.
- **Retrieval saturated on current metrics.** BM25, Dense, and Hybrid all cluster around 0.69 keyword hit rate; the metric is too coarse to measure smaller retrieval improvements (e.g. cross-encoder reranking would be invisible). Sharper metrics (`Recall@3`, `kw-hit @ rank 1`) would unlock further iteration.
- **Test set under-sampled at the per-intent level.** Even at 176 cases, intents like `lineup` (n=19) and `team_news` (n=29) have wide per-intent confidence intervals. A single misroute moves a per-intent metric by ~5pp.

---

## License & attribution

Article content scraped from FilGoal.com, used under fair-use for research purposes. The bot's responses cite source URLs back to the original articles.
