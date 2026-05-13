"""
FilGoalBot — api/main.py
=========================
FastAPI wrapper around FilGoalRAG.

Endpoints:
    POST /ask           — main Q&A endpoint
    GET  /health        — readiness check (runs a real retrieval)
    GET  /              — API info

Run locally:
    uvicorn api.main:app --reload --port 8000

Environment:
    GROQ_API_KEY                — required
    FILGOAL_ALLOWED_ORIGINS     — comma-separated; default = "http://127.0.0.1:7860,http://localhost:7860"
    FILGOAL_RATE_LIMIT          — slowapi limit string; default = "20/minute"
    FILGOAL_LOG_FORMAT          — "text" (default) or "json" for structured logs
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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

ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "FILGOAL_ALLOWED_ORIGINS",
        "http://127.0.0.1:7860,http://localhost:7860",
    ).split(",") if o.strip()
]
RATE_LIMIT = os.getenv("FILGOAL_RATE_LIMIT", "20/minute")


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

limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])

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
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

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
    cached:       bool = False
    # One of: "hit" | "miss" | "skipped_oos" | "skipped_no_chunks"
    #       | "skipped_rate_limit" | "skipped_error"
    cache_reason: str = "miss"
    request_id:   str

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":    "FilGoalBot API",
        "version": "1.1.0",
        "status":  "ready" if rag else "loading",
        "docs":    "/docs",
    }


@app.get("/health")
def health():
    """Deeper readiness check: confirms model is loaded AND a known retrieval
    returns at least one chunk. Catches "FAISS file got corrupted" type
    failures that the previous shallow check missed."""
    if not rag:
        raise HTTPException(status_code=503, detail="Model still loading")
    try:
        chunks = rag.retriever.retrieve("الأهلي", top_k=1)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Retriever error: {e!s}")
    if not chunks:
        raise HTTPException(status_code=503, detail="Retriever returned no results")
    return {"status": "ok", "chunks_loaded": len(rag.retriever.metadata)}


@app.post("/ask", response_model=AskResponse)
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
        result = rag.answer(
            query=req.query,
            **({"filter_type":   req.filter_type}   if req.filter_type   else {}),
            **({"filter_league": req.filter_league} if req.filter_league else {}),
            **({"filter_team":   req.filter_team}   if req.filter_team   else {}),
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
            "request_id":   rid,
            "intent":       result["intent"],
            "model":        result.get("model"),
            "cached":       result.get("cached", False),
            "cache_reason": result.get("cache_reason"),
            "retrieval_ms": result.get("retrieval_ms"),
            "llm_ms":       result.get("llm_ms"),
            "latency_ms":   latency,
            "n_chunks":     result.get("n_chunks"),
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
        cached=result.get("cached", False),
        cache_reason=result.get("cache_reason", "miss"),
        request_id=rid,
    )
