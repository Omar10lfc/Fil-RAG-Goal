"""
FilGoalBot — api/main.py
=========================
FastAPI wrapper around FilGoalRAG.

Endpoints:
    POST /ask           — main Q&A endpoint
    POST /ask/stream    — SSE token-streaming endpoint
    GET  /health        — readiness check (runs a real retrieval)
    GET  /              — API info / static frontend

Run locally:
    uvicorn api.main:app --reload --port 8000

Environment:
    GROQ_API_KEY                — required
    FILGOAL_ALLOWED_ORIGINS     — comma-separated; default = "" (same-origin only)
    FILGOAL_RATE_LIMIT          — slowapi limit string; default = "20/minute"
    FILGOAL_TRUST_PROXY         — "1" (default) honours X-Forwarded-For for
                                  rate-limit keying; "0" uses the socket IP
    FILGOAL_LOG_FORMAT          — "text" (default) or "json" for structured logs
"""

import asyncio
import json
import logging
import os
import pathlib
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.logging_config import configure_logging
from qa_engine.rag_pipeline import FilGoalRAG

load_dotenv()
configure_logging(level=logging.INFO)
log = logging.getLogger("api")

# ─── Config ───────────────────────────────────────────────────────────────────

# Same-origin by default (empty list = no CORS headers). Set
# FILGOAL_ALLOWED_ORIGINS to a comma-separated list when the frontend is
# served from another origin. The old Gradio default (port 7860) is gone —
# the static frontend is served from this same app (see _STATIC_DIR below).
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("FILGOAL_ALLOWED_ORIGINS", "").split(",") if o.strip()
]
RATE_LIMIT = os.getenv("FILGOAL_RATE_LIMIT", "20/minute")


def _is_trust_proxy() -> bool:
    return os.getenv("FILGOAL_TRUST_PROXY", "1").strip() in ("1", "true", "yes")


def _client_key(request: Request) -> str:
    """Rate-limit key: leftmost X-Forwarded-For hop when behind a trusted
    proxy, else the socket IP.

    Behind the HF Space proxy every client shares one socket IP, so keying on
    it turns "20/minute" into an effective GLOBAL cap. With FILGOAL_TRUST_PROXY=1
    (default) we key on the client IP the proxy reports instead.
    Caveat: only trust XFF when a proxy you control sets it — a directly
    exposed server with TRUST_PROXY=1 lets clients spoof arbitrary keys and
    dodge limits. Set FILGOAL_TRUST_PROXY=0 in that deployment.
    """
    if _is_trust_proxy():
        xff = request.headers.get("x-forwarded-for", "")
        if xff.strip():
            return xff.split(",")[0].strip()
    return get_remote_address(request)


def _assert_groq_key_present() -> None:
    """Fail fast at process boot if the Groq key is missing or malformed.
    Checked here in addition to FilGoalRAG.load() so misconfigured deploys
    crash before the FAISS index spends ~5s loading. Never echo the key
    itself in errors."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Refusing to start.")
    if not key.startswith("gsk_") or len(key) < 40:
        raise RuntimeError(
            "GROQ_API_KEY does not look like a valid Groq key "
            "(expected `gsk_` prefix, ≥40 chars). Refusing to start."
        )


_assert_groq_key_present()

# ─── App lifespan ─────────────────────────────────────────────────────────────

rag: FilGoalRAG | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    log.info("Loading FilGoalRAG...")
    rag = FilGoalRAG()
    rag.load()
    log.info("FilGoalRAG ready — API is live")
    yield
    log.info("Shutting down")


# ─── App ──────────────────────────────────────────────────────────────────────

limiter = Limiter(key_func=_client_key, default_limits=[RATE_LIMIT])

app = FastAPI(
    title="FilGoalBot API",
    description="Arabic football Q&A — powered by FilGoal articles + Groq",
    version="1.1.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
# slowapi's handler is typed for its concrete exception class; FastAPI's
# `add_exception_handler` insists on (Request, Exception). The runtime
# contract is correct — Starlette dispatches by exception class — so the
# nominal mismatch is safe to ignore.
# `unused-ignore` is part of the code-list so the ignore is silently
# accepted whether or not the underlying error fires — local dev (no
# slowapi stubs available) sees an arg-type mismatch; CI (with slowapi
# installed) does not. One spelling covers both.
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type, unused-ignore]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


# ─── Schemas ──────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query:         str = Field(..., min_length=2, max_length=500, examples=["من سجل هدف الأهلي أمس؟"])
    filter_league: str | None = Field(None, examples=["egyptian_league", "premier_league"])
    filter_team:   str | None = Field(None, examples=["al_ahly", "zamalek"])
    filter_type:   str | None = Field(None, examples=["lineup", "match_result", "transfer"])
    # Previous turn for follow-ups ("وماذا حدث بعدها؟") — the frontend sends
    # the last {question, answer} exchange here (max 2,000 chars). Folded into
    # the LLM prompt AND the cache key, so follow-ups never hit a stale
    # standalone entry.
    conversation_context: str | None = Field(None, max_length=2000, examples=["السؤال السابق: …\nالإجابة السابقة: …"])

class Source(BaseModel):
    title:        str
    url:          str
    pub_date:     str
    article_type: str
    league:       str
    # New: surface enough metadata that a UI can show *which* article
    # supported each claim and how confident retrieval was. rrf_score is
    # the fused BM25+FAISS RRF value with the recency multiplier applied
    # (see retrieval/hybrid_retriever.retrieve).
    chunk_id:     str = ""
    rrf_score:    float = 0.0

class AskResponse(BaseModel):
    answer:       str
    intent:       str
    sources:      list[Source]
    latency_ms:   int
    # Latency split — retrieval and LLM are the two big-ticket costs, and
    # are useful to separate when debugging a slow tail (slow FAISS read
    # vs. slow Groq response).
    retrieval_ms: int = 0
    llm_ms:       int = 0
    model:        str | None = None
    # True if the request was originally routed to the 70B but rate-limited,
    # and the pipeline automatically retried on the 8B. Useful for the
    # frontend to render a "answered with fallback model" indicator.
    model_fallback: bool = False
    cached:       bool = False
    # One of: "hit" | "miss" | "skipped_oos" | "skipped_no_chunks"
    #       | "skipped_rate_limit" | "skipped_error"
    cache_reason: str = "miss"
    request_id:   str

# ─── Endpoints ────────────────────────────────────────────────────────────────

# Resolve the static frontend directory relative to this file.
_STATIC_DIR = pathlib.Path(__file__).resolve().parent.parent / "frontend" / "static"


@app.get("/")
def root():
    """Serve the custom frontend if available, otherwise return API info JSON."""
    index = _STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(str(index))
    return {
        "name":    "FilGoalBot API",
        "version": "1.1.0",
        "status":  "ready" if rag else "loading",
        "docs":    "/docs",
    }


@app.get("/health")
def health():
    """Deeper readiness check: confirms model is loaded, the retriever works,
    AND the Groq client is initialised. Catches "FAISS file got corrupted",
    expired API keys, and other config failures before users hit them."""
    if not rag:
        raise HTTPException(status_code=503, detail="Model still loading")
    if rag.groq is None:
        raise HTTPException(status_code=503, detail="Groq client not initialized")
    try:
        chunks = rag.retriever.retrieve("الأهلي", top_k=1)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Retriever error: {e!s}")
    if not chunks:
        raise HTTPException(status_code=503, detail="Retriever returned no results")
    return {
        "status": "ok",
        "chunks_loaded": len(rag.retriever.metadata),
        "groq_ready": True,
    }


@app.post(
    "/ask",
    response_model=AskResponse,
    responses={
        429: {"description": "Rate limit exceeded — retry after the window resets"},
        503: {"description": "Model still loading or retriever unavailable"},
        500: {"description": "Internal error generating answer (Groq API or pipeline failure)"},
    },
)
@limiter.limit(RATE_LIMIT)
async def ask(req: AskRequest, request: Request):
    if not rag:
        raise HTTPException(status_code=503, detail="Model still loading")

    rid = request.state.request_id
    log.info(
        "ask received",
        extra={"request_id": rid, "query_len": len(req.query)},
    )

    start = time.monotonic()
    try:
        # rag.answer() is fully synchronous (FAISS retrieval + blocking Groq
        # HTTP call, seconds-long). Running it inline would block the event
        # loop and stall every concurrent request, so push it to a worker
        # thread and keep the loop free.
        result = await asyncio.to_thread(
            rag.answer,
            query=req.query,
            **({"filter_type":   req.filter_type}   if req.filter_type   else {}),
            **({"filter_league": req.filter_league} if req.filter_league else {}),
            **({"filter_team":   req.filter_team}   if req.filter_team   else {}),
            **({"conversation_context": req.conversation_context} if req.conversation_context else {}),
        )
    except Exception as e:
        # Log type + message only, NOT the full traceback — Groq SDK frames
        # carry the client object whose locals include the api_key. We have
        # a request id for correlation, so a stack trace isn't worth the
        # credential-disclosure risk.
        log.error(
            f"RAG error: {type(e).__name__}: {e}",
            extra={"request_id": rid},
        )
        raise HTTPException(status_code=500, detail="Internal error generating answer")

    latency = int((time.monotonic() - start) * 1000)
    log.info(
        "ask answered",
        extra={
            "request_id":     rid,
            "intent":         result["intent"],
            "model":          result.get("model"),
            "model_fallback": result.get("model_fallback", False),
            "cached":         result.get("cached", False),
            "cache_reason":   result.get("cache_reason"),
            "retrieval_ms":   result.get("retrieval_ms"),
            "llm_ms":         result.get("llm_ms"),
            "latency_ms":     latency,
            "n_chunks":       result.get("n_chunks"),
        },
    )

    return AskResponse(
        answer=result["answer"],
        intent=result["intent"],
        sources=[
            Source(**{k: s.get(k, "" if k != "rrf_score" else 0.0)
                      for k in ("title", "url", "pub_date", "article_type",
                                "league", "chunk_id", "rrf_score")})
            for s in result["sources"]
        ],
        latency_ms=latency,
        retrieval_ms=result.get("retrieval_ms", 0),
        llm_ms=result.get("llm_ms", 0),
        model=result.get("model"),
        model_fallback=result.get("model_fallback", False),
        cached=result.get("cached", False),
        cache_reason=result.get("cache_reason", "miss"),
        request_id=rid,
    )


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post(
    "/ask/stream",
    responses={
        429: {"description": "Rate limit exceeded — retry after the window resets"},
        503: {"description": "Model still loading or retriever unavailable"},
        500: {"description": "Internal error generating answer (Groq API or pipeline failure)"},
    },
)
@limiter.limit(RATE_LIMIT)
async def ask_stream(req: AskRequest, request: Request):
    """SSE token stream: `meta` (intent/model/sources) → `delta` per token →
    `done` (latency, cache_reason, model_fallback) / `error`.

    The blocking rag.answer_stream() generator runs in a worker thread and
    pushes updates through an asyncio.Queue, so the event loop never blocks
    and concurrent /ask requests keep flowing.
    """
    if not rag:
        raise HTTPException(status_code=503, detail="Model still loading")

    rid = request.state.request_id
    log.info("ask_stream received", extra={"request_id": rid, "query_len": len(req.query)})
    start = time.monotonic()

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    stream_rag = rag  # local bind for the worker closure
    q = req.query
    filters: dict = {}
    if req.filter_type:   filters["filter_type"]   = req.filter_type
    if req.filter_league: filters["filter_league"] = req.filter_league
    if req.filter_team:   filters["filter_team"]   = req.filter_team
    conv = req.conversation_context

    def _pump() -> None:
        try:
            for update in stream_rag.answer_stream(query=q, **filters,
                                                   conversation_context=conv):
                asyncio.run_coroutine_threadsafe(queue.put(("update", update)), loop).result()
        except Exception as e:
            log.error(f"RAG stream error: {type(e).__name__}: {e}", extra={"request_id": rid})
            asyncio.run_coroutine_threadsafe(
                queue.put(("error", "Internal error generating answer")), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(("end", None)), loop).result()

    pump_task = loop.run_in_executor(None, _pump)

    async def _events():
        meta_sent = False
        sent_len = 0
        last: dict = {}
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "error":
                    yield _sse("error", {"detail": payload, "request_id": rid})
                    return
                if kind == "end":
                    latency = int((time.monotonic() - start) * 1000)
                    yield _sse("done", {
                        "latency_ms":     latency,
                        "retrieval_ms":   last.get("retrieval_ms", 0),
                        "llm_ms":         last.get("llm_ms", 0),
                        "model":          last.get("model"),
                        "model_fallback": last.get("model_fallback", False),
                        "cached":         last.get("cached", False),
                        "cache_reason":   last.get("cache_reason", "miss"),
                        "request_id":     rid,
                    })
                    log.info("ask_stream answered",
                             extra={"request_id": rid, "intent": last.get("intent"),
                                    "cached": last.get("cached", False)})
                    return
                # kind == "update"
                last = payload
                if not meta_sent:
                    meta_sent = True
                    yield _sse("meta", {
                        "intent":  payload["intent"],
                        "model":   payload.get("model"),
                        "sources": payload.get("sources", []),
                        "request_id": rid,
                    })
                text = payload.get("answer", "")
                if len(text) > sent_len:
                    yield _sse("delta", {"text": text[sent_len:]})
                    sent_len = len(text)
        finally:
            await pump_task

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-ID": rid,
        },
    )


# ── Static files (must be last — catches /static/* paths) ─────────────────────
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
