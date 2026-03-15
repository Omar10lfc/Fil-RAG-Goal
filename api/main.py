"""
FilGoalBot — api/main.py
=========================
FastAPI wrapper around FilGoalRAG.

Endpoints:
    POST /ask           — main Q&A endpoint
    GET  /health        — liveness check
    GET  /              — API info

Run locally:
    uvicorn api.main:app --reload --port 8000

Install:
    pip install fastapi uvicorn python-dotenv
"""

import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from qa_engine.rag_pipeline import FilGoalRAG

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("api")

# ─── App lifespan — load model once at startup ────────────────────────────────

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

app = FastAPI(
    title="FilGoalBot API",
    description="Arabic football Q&A — powered by FilGoal articles + Groq",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Schemas ──────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500, examples=["من سجل هدف الأهلي أمس؟"])
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

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":    "FilGoalBot API",
        "version": "1.0.0",
        "status":  "ready" if rag else "loading",
        "docs":    "/docs",
    }


@app.get("/health")
def health():
    if not rag:
        raise HTTPException(status_code=503, detail="Model still loading")
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request):
    if not rag:
        raise HTTPException(status_code=503, detail="Model still loading")

    log.info(f"[{request.client.host}] /ask — '{req.query[:80]}'")

    start = time.monotonic()
    try:
      result = rag.answer(
        query=req.query,
        # Pass caller filters only if explicitly provided,
        # they override the intent router
        **({"filter_type": req.filter_type} if req.filter_type else {}),
        **({"filter_league": req.filter_league} if req.filter_league else {}),
        **({"filter_team": req.filter_team} if req.filter_team else {}),
    )
    except Exception as e:
        log.error(f"RAG error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error generating answer")

    latency = int((time.monotonic() - start) * 1000)
    log.info(f"  intent={result['intent']} | {latency}ms")

    return AskResponse(
        answer=result["answer"],
        intent=result["intent"],
        sources=[Source(**s) for s in result["sources"]],
        latency_ms=latency,
    )