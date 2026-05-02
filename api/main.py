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

from qa_engine.rag_pipeline import FilGoalRAG

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("api")

# ─── Config ───────────────────────────────────────────────────────────────────

ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "FILGOAL_ALLOWED_ORIGINS",
        "http://127.0.0.1:7860,http://localhost:7860",
    ).split(",") if o.strip()
]
RATE_LIMIT = os.getenv("FILGOAL_RATE_LIMIT", "20/minute")

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
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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

class AskResponse(BaseModel):
    answer:       str
    intent:       str
    sources:      list[Source]
    latency_ms:   int
    model:        str | None = None
    cached:       bool = False
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
    log.info(f"[{rid}] /ask — '{req.query[:80]}'")

    start = time.monotonic()
    try:
        result = rag.answer(
            query=req.query,
            **({"filter_type":   req.filter_type}   if req.filter_type   else {}),
            **({"filter_league": req.filter_league} if req.filter_league else {}),
            **({"filter_team":   req.filter_team}   if req.filter_team   else {}),
        )
    except Exception as e:
        log.error(f"[{rid}] RAG error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error generating answer")

    latency = int((time.monotonic() - start) * 1000)
    log.info(f"[{rid}] intent={result['intent']} model={result.get('model')} cached={result.get('cached')} {latency}ms")

    return AskResponse(
        answer=result["answer"],
        intent=result["intent"],
        sources=[Source(**{k: s[k] for k in ("title", "url", "pub_date", "article_type", "league")}) for s in result["sources"]],
        latency_ms=latency,
        model=result.get("model"),
        cached=result.get("cached", False),
        request_id=rid,
    )
