"""
FilGoalBot Hybrid Retriever
============================
BM25 (sparse) + FAISS (dense) fused with weighted RRF, with optional:
  - Arabic clitic stripping in the BM25 tokenizer
  - Date-decay recency boost (configurable half-life)
  - Pool expansion when metadata filters are active

Dependencies:
    pip install rank-bm25
"""

import json
import logging
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

log = logging.getLogger("retriever")

FAISS_DIR    = Path("faiss_index")
INDEX_FILE   = FAISS_DIR / "index.bin"
META_FILE    = FAISS_DIR / "metadata.jsonl"
CONFIG_FILE  = FAISS_DIR / "config.json"

# RRF parameters
RRF_K         = 60
DENSE_WEIGHT  = 0.7   # dense beat sparse on every per-intent metric in the ablation
SPARSE_WEIGHT = 0.3

# Recency boost — football news is time-sensitive. Score multiplier:
#   recent (today)        → ~1.10
#   1 month old           → ~1.05
#   1 year old            → ~1.00 (no boost)
RECENCY_HALF_LIFE_DAYS = 30
RECENCY_MAX_BOOST      = 0.10

TOP_K_BM25       = 50
TOP_K_DENSE      = 50
TOP_K_FINAL      = 8
FILTERED_POOL_X  = 4  # when filters are active, pull this many times more candidates


# ─── Arabic normalisation ────────────────────────────────────────────────────

_HARAKAT = re.compile(r'[ً-ٰٟـ]')
# Arabic clitics commonly attached to the next word: و ف ب ل ك + ال
# Stripping these before BM25 reduces vocab inflation and improves IDF.
_CLITICS = re.compile(r'^(?:و|ف|ب|ل|ك)?(?:ال)?')


def _normalize_ar(text: str) -> str:
    text = _HARAKAT.sub('', text)
    for frm, to in [('[أإآٱ]', 'ا'), ('ى', 'ي'), ('ة', 'ه'), ('ؤ', 'و')]:
        text = re.sub(frm, to, text)
    return text


def _strip_clitics(token: str) -> str:
    """Remove leading w/f/b/l/k + optional al- definite article. Defensive: if
    stripping leaves a stub of <2 chars, keep the original token."""
    stripped = _CLITICS.sub('', token, count=1)
    return stripped if len(stripped) >= 2 else token


def _tokenize(text: str) -> list[str]:
    """Normalise → split → strip clitics → drop short stop-tokens."""
    tokens = []
    for raw in _normalize_ar(text).split():
        if len(raw) < 2:
            continue
        tokens.append(_strip_clitics(raw))
    return tokens


# ─── Dialect → MSA query expansion (BM25 only) ─────────────────────────────
# Corpus is MSA, queries are often Egyptian dialect. Appended to BM25 query
# tokens only — dense side untouched so this can't hurt semantic matching.
# Append-only (never replaces) to avoid deleting the original signal.
DIALECT_EXPANSIONS = {
    "سكور": "سجل نتيجة",
    "اتعادل": "تعادل",
    "يتعادل": "تعادل",
    "بكام": "بكم",
    "امبارح": "امس",
    "النهاردة": "اليوم",
    "عايز": "يريد",
    "ازاي": "كيف",
    "ليه": "لماذا",
    "مين": "من",
    "امتي": "متى",
    "فين": "اين",
    "كسب": "فاز",
    "يكسب": "يفوز",
    "جون": "هدف",
    "اجوان": "اهداف",
}


def _dialect_lookup() -> dict[str, list[str]]:
    """Normalised single-token lookup built from DIALECT_EXPANSIONS.

    Both query tokens and keys go through _tokenize() (which strips clitics),
    so variants like السكور/سكور collapse to one entry — first wins, which
    also prevents double-adding the same MSA tokens.
    """
    lookup: dict[str, list[str]] = {}
    for raw, msa in DIALECT_EXPANSIONS.items():
        keys = _tokenize(raw)
        if len(keys) == 1:
            lookup.setdefault(keys[0], _tokenize(msa))
    return lookup


_DIALECT_LOOKUP = _dialect_lookup()


def _expand_dialect(tokens: list[str]) -> list[str]:
    """Append MSA equivalents for known Egyptian-dialect tokens.

    Append-only (never replaces) and deduped — each MSA token is added at
    most once so colliding variants can't double-weight the BM25 query.
    """
    expanded = list(tokens)
    seen = set(tokens)
    for tok in tokens:
        for add in _DIALECT_LOOKUP.get(tok, ()):
            if add not in seen:
                seen.add(add)
                expanded.append(add)
    return expanded


# ─── Recency helper ──────────────────────────────────────────────────────────

def _recency_multiplier(pub_date: str, now: datetime | None = None) -> float:
    """Returns a multiplier in [1.0, 1 + RECENCY_MAX_BOOST] based on age.

    Uses an exponential decay so today is fully boosted and old articles are
    barely boosted. Returns 1.0 if pub_date is missing or unparseable."""
    if not pub_date:
        return 1.0
    try:
        published = datetime.fromisoformat(pub_date[:10]).replace(tzinfo=timezone.utc)
    except ValueError:
        return 1.0
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    age_days = max(0, (now - published).days)
    decay = math.exp(-age_days / RECENCY_HALF_LIFE_DAYS)
    return 1.0 + RECENCY_MAX_BOOST * decay


# ─── Hybrid Retriever ────────────────────────────────────────────────────────

class FilGoalRetriever:
    def __init__(self):
        self.metadata: list[dict] = []
        self.texts:    list[str]  = []
        self.bm25:     BM25Okapi | None = None
        self.index     = None
        self.model     = None
        self.dim: int  = 768

    def load(self):
        """Load FAISS index, metadata, and build BM25Okapi."""
        import faiss
        from sentence_transformers import SentenceTransformer

        config = json.loads(CONFIG_FILE.read_text())
        self.dim = config['dim']
        model_name = config['model_name']

        log.info(f"Loading FAISS index from {INDEX_FILE}...")
        self.index = faiss.read_index(str(INDEX_FILE))
        log.info(f"  {self.index.ntotal} vectors loaded")

        log.info(f"Loading metadata from {META_FILE}...")
        with open(META_FILE, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))
        self.texts = [m['text'] for m in self.metadata]
        log.info(f"  {len(self.metadata)} chunks loaded")

        log.info("Building BM25Okapi index...")
        tokenized_corpus = [_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        log.info("  BM25Okapi ready")

        log.info(f"Loading sentence model: {model_name}")
        self.model = SentenceTransformer(model_name)

        log.info("✅ Retriever ready")

    def _dense_search(self, query: str, top_k: int = TOP_K_DENSE) -> list[tuple[int, float]]:
        if self.index is None or self.model is None:    # ablation: dense disabled
            return []
        q_emb = self.model.encode(
            ["query: " + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        D, I = self.index.search(q_emb.astype(np.float32), k=top_k)
        return list(zip(I[0].tolist(), D[0].tolist()))

    def _sparse_search(self, query: str, top_k: int = TOP_K_BM25) -> list[tuple[int, float]]:
        if self.bm25 is None:                           # ablation: BM25 disabled
            return []
        query_tokens = _expand_dialect(_tokenize(query))
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def _rrf_fuse(
        self,
        bm25_results:  list[tuple[int, float]],
        dense_results: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """Weighted RRF — dense outperforms sparse in the ablation, so it
        carries more weight in the fused score."""
        scores: dict[int, float] = defaultdict(float)
        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] += SPARSE_WEIGHT * (1.0 / (RRF_K + rank + 1))
        for rank, (idx, _) in enumerate(dense_results):
            scores[idx] += DENSE_WEIGHT * (1.0 / (RRF_K + rank + 1))
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_FINAL,
        filter_type:   str | None = None,
        filter_league: str | None = None,
        filter_team:   str | None = None,
        date_from:     str | None = None,   # ISO YYYY-MM-DD, inclusive
        date_to:       str | None = None,   # ISO YYYY-MM-DD, inclusive
    ) -> list[dict]:
        """Hybrid search with optional metadata filters and recency boost.
        Returns top_k chunk dicts, deduplicated by article_id."""
        # When filters are active, pull a wider initial pool so filtering
        # doesn't starve top_k.
        filters_active = any([filter_type, filter_league, filter_team,
                              date_from, date_to])
        bm25_n  = TOP_K_BM25  * (FILTERED_POOL_X if filters_active else 1)
        dense_n = TOP_K_DENSE * (FILTERED_POOL_X if filters_active else 1)

        bm25_results  = self._sparse_search(query, top_k=bm25_n)
        dense_results = self._dense_search(query,  top_k=dense_n)
        fused         = self._rrf_fuse(bm25_results, dense_results)

        # Apply recency multiplier and re-sort.
        now = datetime.now(timezone.utc)
        boosted: list[tuple[int, float]] = []
        for idx, score in fused:
            if idx >= len(self.metadata):
                continue
            mult = _recency_multiplier(self.metadata[idx].get("pub_date", ""), now=now)
            boosted.append((idx, score * mult))
        boosted.sort(key=lambda x: x[1], reverse=True)

        results: list[dict] = []
        seen_article_ids: set[str] = set()
        for idx, score in boosted:
            chunk = self.metadata[idx]

            if filter_type   and chunk.get('article_type') != filter_type:
                continue
            if filter_league and chunk.get('league') != filter_league:
                continue
            if filter_team   and filter_team not in chunk.get('teams', []):
                continue
            # Date-window filter (match_result "امبارح" queries). Chunks with
            # missing/unparseable dates can't satisfy a dated question, so they
            # are skipped only while a window is active.
            if date_from or date_to:
                day = str(chunk.get('pub_date', ''))[:10]
                if not day or (date_from and day < date_from) or (date_to and day > date_to):
                    continue

            aid = chunk['article_id']
            if aid in seen_article_ids:
                continue
            seen_article_ids.add(aid)

            results.append({**chunk, '_rrf_score': score})
            if len(results) >= top_k:
                break

        return results
