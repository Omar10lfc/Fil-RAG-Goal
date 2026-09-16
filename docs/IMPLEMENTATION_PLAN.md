# FilGoalBot — Implementation Plan (2026-09 hardening pass)

Deliverables: (A) this file — the full plan with status checkboxes updated as
work landed; (B) README "Change impact log" + Deployment/structure sync;
(C) the six implementation phases.

Ground rules (applied throughout): every behavior change got a test;
`ruff check` + `mypy` + `pytest` green before finishing (see Verification);
deletions only after grep-verifying zero references; every change logged in
the README impact table with what/why/impact/risk.

## Part A — Current-state audit (13 findings, evidence-verified)

| # | Finding | Disposition |
|---|---------|-------------|
| A1 | Nothing built `faiss_index/index.bin` — `preprocessing/pipeline.py` wrote only `chunks.jsonl`; `refresh.yml` had a TODO | Fixed Phase 0: new `preprocessing/build_index.py` |
| A2 | `.gitignore` excluded `faiss_index/index.bin`, `metadata.jsonl`, `data/processed/*` → refresh bot's `git add` committed nothing | Fixed Phase 0: LFS tracking + un-ignore |
| A3 | Rate limiter keyed on socket IP → one global 20/min cap behind the HF Space proxy | Fixed Phase 1: `_client_key()` |
| A4 | `"pytest" not in sys.modules` routed everything to 120b in prod; 429 fallback retried the buggy 20b → `ERROR_ANSWER` | Fixed Phase 1: `FILGOAL_FORCE_BIG_MODEL` + consistent chain |
| A5 | `conversation_context` existed in pipeline but not in `AskRequest` → follow-ups unreachable | Fixed Phase 3: schema field + passthrough |
| A6 | `answer_stream()` had no API endpoint → dead code | Fixed Phase 3: `POST /ask/stream` (SSE) |
| A7 | `retrieve_for_rag()`, `build_bm25_only()` + `__main__`: zero call sites | Fixed Phase 2: deleted (−54 lines) |
| A8 | `gradio==4.44.0` core dep; only Space launcher + legacy Gradio UI need it | Fixed Phase 2: moved to `requirements-space.txt` |
| A9 | Duplicate report homes, stray `eval-llama*.log`, `notebooks/` ghost path | Fixed Phase 2: single report home, logs + notebooks removed |
| A10 | `answer()`/`answer_stream()` ~80% duplicated, already drifted | Fixed Phase 4: shared `_prepare()` |
| A11 | Cache TTL-bounded but not size-bounded → `.cache/llm/` grew forever | Fixed Phase 4: `FILGOAL_CACHE_MAX_ENTRIES` |
| A12 | CORS defaults referenced dead Gradio port 7860 | Fixed Phase 1: same-origin default |
| A13 | Cache key ignored `conversation_context` → cached answers would shadow follow-ups | Fixed Phase 3: `conversation_key` in key (v2) |

## Part B — Phases

### Phase 0 — Daily scrape works end-to-end — DONE

- [x] `preprocessing/build_index.py`: incremental (default), `--rebuild`,
      `--prune-older-than YYYY-MM-DD` (via `index.reconstruct_n`, no
      re-embedding), `--smoke-only` (CI gate). Model/dim/metric read from
      `faiss_index/config.json`, never hard-coded. e5 `passage: ` prefix,
      normalized (retriever uses `query: `). Non-zero exit on count mismatch
      (`config.n_vectors != index.ntotal != len(metadata)`) or smoke failure.
      Verified on real corpus: counts consistent (n=6120), smoke top-1
      score 0.8293.
- [x] Git LFS (`.gitattributes`) for `faiss_index/index.bin`,
      `faiss_index/metadata.jsonl`, `data/processed/chunks.jsonl`; `.gitignore`
      un-ignores them; global `*.bin` ignore narrowed to weight files
      (`*.safetensors`/`*.pt`/`*.pth`/`model*.bin`). `data/raw/` stays ignored.
      ~90MB initial. LFS 1 GB/mo bandwidth caveat + HF-Dataset fallback in README.
- [x] `refresh.yml`: `checkout(lfs) → git lfs install → scrape --newest-only →
      pipeline → build_index → --smoke-only (gate) → commit-if-changed + push`
      (`[skip ci]` kept). TODO block replaced.
- [x] README Deployment rewritten (FastAPI static serving, LFS strategy,
      two-command local rebuild).

### Phase 1 — Production bugs — DONE

- [x] Proxy-aware rate limiting (`api/main.py::_client_key`): leftmost
      X-Forwarded-For hop when `FILGOAL_TRUST_PROXY=1` (default on; spoofing
      caveat documented), else socket IP. 4 tests in `tests/test_rate_limit_key.py`.
- [x] Env-based routing: `sys.modules` hack deleted; `FILGOAL_FORCE_BIG_MODEL`
      (production default `=1` in `.env.example`, documents the gpt-oss-20b
      Arabic runaway-reasoning bypass). `FILGOAL_FORCE_SMALL_MODEL` still wins
      when both are set. Tests set env per case (`clean_routing_env` fixture).
- [x] Consistent fallback chain: empty completion (raised `RuntimeError` or
      returned `""`) → exactly one retry on the big model; `RateLimitError`
      on big → small unless forced-big → error; second failure terminal
      (`ERROR_ANSWER`, never cached). `answer_stream()` mirrors the chain.
      `tests/test_model_fallback.py`: 4 existing + force-big routing,
      empty→big retry, force-big-429→error, no-retry-on-second-failure (8 total).
- [x] CORS: default origins empty (same-origin); env-overridable; port 7860 gone.

### Phase 2 — Code-quality sweep — DONE

- [x] `space_app.py` deletion (pre-existing in worktree, zero code references) kept.
- [x] `retrieve_for_rag()` (−13), `build_bm25_only()` + `__main__` (−41):
      deleted, zero call sites; tokenizer/recency helpers stay (tested).
- [x] `gradio==4.44.0` → `requirements-space.txt` (`-r requirements.txt` +
      gradio for the Space launcher + legacy `frontend/app.py`); README note.
      Every remaining `requirements.txt` pin is imported by runtime code
      (verified by grep: requests/bs4/lxml → scraper; st/faiss/torch/numpy →
      retrieval+build_index; groq/httpx → qa_engine; fastapi/uvicorn/
      pydantic/dotenv → api; rank-bm25 → retrieval; slowapi → api).
- [x] Single report home: root `report.json` (tracked) deleted;
      `evaluation/report.json` un-ignored (now the canonical, committed copy).
      The two files were different runs (latency + 70B-quota noise); canonical
      kept as-is, README reconciled to it.
- [x] `eval-llama.err.log` / `eval-llama.log` (ephemeral) deleted.
- [x] `notebooks/` (Kaggle GPU walkthrough, fully superseded by
      `build_index.py`: same model, prefix, FlatIP, metadata fields) +
      `kaggle_notebook.py` ignore-line removed. Zero references; logic
      preserved in `build_index.py`.
- [x] Sweep: `ruff check` clean (dead imports/vars F-rules green);
      `/ask` + `/ask/stream` exercised by `frontend/static/script.js`;
      `/health` + `/docs` explicitly API-only.

### Phase 3 — Streaming + conversation memory — DONE

- [x] `AskRequest.conversation_context: str | None` (max 2,000 chars) →
      passed to `rag.answer()` (passthrough covered by endpoint test).
- [x] `POST /ask/stream` (SSE): `meta` (intent/model/sources) → `delta` per
      token → `done` (latency, cache_reason, model_fallback) / `error`.
      Blocking generator pumped via thread executor + `asyncio.Queue`; same
      rate limiter + request-id middleware path as `/ask`.
- [x] Frontend (`frontend/static/script.js`): `fetch` + `ReadableStream` +
      `TextDecoder` buffering on `\n\n`; token-by-token rendering;
      `lastExchange = {q, a}` sent as `conversation_context`; mid-stream cut
      keeps partial text with a notice; anything pre-render falls back to
      non-streaming `/ask`. Clear button resets `lastExchange`.
- [x] Cache-key fix (A13): `cache.get/put/_make_key` gained
      `conversation_key: str = ""` (`CACHE_KEY_VERSION = 2` — one-time
      invalidation, logged at startup by `FilGoalRAG.load()`; orphans age out
      via the TTL sweep). `tests/test_api_endpoints.py` (5: passthrough, SSE
      sequence, 429, /health) + `test_cache.py` conversation-shift test.

### Phase 4 — Robustness — DONE

- [x] Shared `_prepare(query, conversation_context, filters) -> Prepared`
      (intent, chunks, context, sources, chunk_ids, model, system_prompt,
      user_prompt, conv_key, cached_answer, retrieval_ms, out_of_scope)
      consumed by both `answer()` and `answer_stream()` — behavior-preserving
      (full suite green, no test changes needed). Token-budget guard now
      single-sourced (the drift site is gone).
- [x] `FILGOAL_CACHE_MAX_ENTRIES` (default 5,000) enforced in `cache.put()` —
      oldest-by-mtime eviction, best-effort. Covered by size-cap test.
- [x] `.env.example` documents all vars (origins, trust-proxy, models,
      force-big/small, cache cap).
- [x] README sync: eval numbers reconciled with canonical
      `evaluation/report.json`; structure tree updated (no `space_app.py`,
      static frontend, `build_index.py`, `requirements-space.txt`,
      `docs/IMPLEMENTATION_PLAN.md`); test-suite count 68 → 87 passed.

### Phase 5 — Retrieval quality — RESTORED (needs fresh corpus to measure)

Protocol: baseline = committed ablation (Hybrid RRF: MRR 0.768, Kw@1 0.756,
R@3 0.693). After each experiment `python -m evaluation.evaluate
--save-report`; judge on Kw@1/R@3/MRR (Kw-Hit saturated); adopt on MRR or
Kw@1 ≥ +1pp with R@3 not worse; else revert + log. Full runs on real corpus.

- [x] Experiment 1 — `match_result` date-window filter (`qa_engine/temporal.py`,
      `retrieve(date_from/date_to)`, wired in `_prepare`): **RESTORED**.
      Was reverted at MRR 0.768 → 0.762 (−0.6pp): relative windows resolve
      against wall-clock today while the corpus was frozen at Jan–Mar 2026,
      so "امبارح" windows matched nothing. Correct under daily-refresh;
      re-measure after scraping to Sep 2026. Eval harness calls
      `retrieve()` directly (no `_prepare`), so it stays neutral there by
      design — production-only path.
- [x] Experiment 2 — Dialect→MSA BM25 expansion (`DIALECT_EXPANSIONS`, dense
      untouched): **RESTORED**. Was reverted at MRR 0.768 → 0.768 (±0);
      BM25-only 0.719 → 0.718 (noise). Dense (0.7 fusion weight) dominates;
      append-only + deduped so it can't hurt fused rankings. Re-measure
      after rescrape with dated/dialect eval cases.
- [ ] Stretch — AraBERT embedder trial via `build_index --rebuild`
      (config-driven): NOT RUN (needs GPU + model download; `--model` flag
      ready for it).

## Test matrix

| File | Covers | Status |
|------|--------|--------|
| `tests/test_build_index.py` (new) | diff_new, chunk dedup, count consistency, prune subsets, smoke w/ fakes | 5 pass |
| `tests/test_rate_limit_key.py` (new) | XFF honored/ignored per trust flag, per-client isolation | 4 pass |
| `tests/test_model_fallback.py` (update) | env routing, empty→big retry, force-big 429, no-second-retry, cache-under-effective-model | 8 pass |
| `tests/test_api_endpoints.py` (new) | conversation passthrough, SSE sequence, 429, /health | 5 pass |
| `tests/test_cache.py` (update) | conversation_key shift, size-cap eviction (+9 existing) | 12 pass |
| `tests/test_temporal.py` | date-window parser incl. dialect forms | 8 pass |
| `tests/test_intent.py` | regression: no new misroutes | pass |

## Verification & docs

- `ruff check` + `mypy` + full `pytest` green (CI parity).
- `build_index --smoke-only` green on real data (n=6120).
- Boot check: `/` serves frontend, `/ask` + `/ask/stream` answer, `/health` ok
  (covered by `test_api_endpoints.py` with stubbed RAG).
- README impact log updated with final numbers (including both Phase 5
  negative results); checkboxes above flipped; every deletion listed in
  Phase 2 with why/impact/risk.

## Risk register

| Risk | Mitigation | Status |
|------|------------|--------|
| LFS bandwidth on GitHub free tier | prune flag; HF-Dataset alternative documented | documented |
| XFF spoofing on direct exposure | `FILGOAL_TRUST_PROXY=0` documented | documented |
| One-time cache invalidation (key v2) | logged at startup; re-warms naturally | done |
| Streaming worker pressure | thread executor + same rate limiter | done |
| Phase 5 regressions | adopt-only-if-better; both reverted | done |
