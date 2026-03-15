"""
FilGoalBot Hybrid Retriever
============================
Combines BM25 (sparse) + FAISS (dense) with Reciprocal Rank Fusion.

Dependencies:
    pip install rank-bm25
"""

import re
import json
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
from rank_bm25 import BM25Okapi

log = logging.getLogger("retriever")

FAISS_DIR    = Path("faiss_index")
INDEX_FILE   = FAISS_DIR / "index.bin"
META_FILE    = FAISS_DIR / "metadata.jsonl"
CONFIG_FILE  = FAISS_DIR / "config.json"

K           = 60   # RRF constant
TOP_K_BM25  = 50
TOP_K_DENSE = 50
TOP_K_FINAL = 8


# ─── Arabic normalisation (mirrors pipeline.py) ───────────────────────────────

_HARAKAT = re.compile(r'[\u064B-\u065F\u0670\u0640]')

def _normalize_ar(text: str) -> str:
    text = _HARAKAT.sub('', text)
    for frm, to in [('[أإآٱ]', 'ا'), ('ى', 'ي'), ('ة', 'ه'), ('ؤ', 'و')]:
        text = re.sub(frm, to, text)
    return text

def _tokenize(text: str) -> list[str]:
    """Normalise + split, drop tokens shorter than 2 chars (prepositions etc.)."""
    return [t for t in _normalize_ar(text).split() if len(t) >= 2]


# ─── Hybrid Retriever ─────────────────────────────────────────────────────────

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

        # Load config
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

        # BM25Okapi receives a list of token lists — we normalise + tokenize here
        # so Arabic alef variants, tashkeel etc. are consistent with query tokenisation
        log.info("Building BM25Okapi index...")
        tokenized_corpus = [_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        log.info("  BM25Okapi ready")

        log.info(f"Loading sentence model: {model_name}")
        self.model = SentenceTransformer(model_name)

        log.info("✅ Retriever ready")

    def _dense_search(self, query: str, top_k: int = TOP_K_DENSE) -> list[tuple[int, float]]:
        if self.index is None or self.model is None:   # ablation: dense disabled
            return []
        q_emb = self.model.encode(
            ["query: " + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        D, I = self.index.search(q_emb.astype(np.float32), k=top_k)
        return list(zip(I[0].tolist(), D[0].tolist()))

    def _sparse_search(self, query: str, top_k: int = TOP_K_BM25) -> list[tuple[int, float]]:
        if self.bm25 is None:                          # ablation: BM25 disabled
            return []
        query_tokens = _tokenize(_normalize_ar(query))
        scores = self.bm25.get_scores(query_tokens)          # NumPy array, all docs
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def _rrf_fuse(
        self,
        bm25_results:  list[tuple[int, float]],
        dense_results: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] += 1.0 / (K + rank + 1)
        for rank, (idx, _) in enumerate(dense_results):
            scores[idx] += 1.0 / (K + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_FINAL,
        filter_type:   str | None = None,
        filter_league: str | None = None,
        filter_team:   str | None = None,
    ) -> list[dict]:
        """
        Hybrid search with optional metadata filters.
        Returns top_k chunk dicts, deduplicated by article_id.
        """
        bm25_results  = self._sparse_search(query, top_k=TOP_K_BM25)
        dense_results = self._dense_search(query,  top_k=TOP_K_DENSE)
        fused         = self._rrf_fuse(bm25_results, dense_results)

        results = []
        seen_article_ids = set()
        for idx, score in fused:
            if idx >= len(self.metadata):
                continue
            chunk = self.metadata[idx]

            if filter_type   and chunk.get('article_type') != filter_type:
                continue
            if filter_league and chunk.get('league') != filter_league:
                continue
            if filter_team   and filter_team not in chunk.get('teams', []):
                continue

            aid = chunk['article_id']
            if aid in seen_article_ids:
                continue
            seen_article_ids.add(aid)

            results.append({**chunk, '_rrf_score': score})
            if len(results) >= top_k:
                break

        return results

    def retrieve_for_rag(self, query: str, top_k: int = TOP_K_FINAL, **kwargs) -> str:
        """
        Returns a single formatted context string ready for an LLM prompt.
        Each chunk: [N] <title> (<date>) — <league>\n<body_clean>
        """
        hits = self.retrieve(query, top_k=top_k, **kwargs)
        parts = []
        for i, h in enumerate(hits, 1):
            date   = h.get('pub_date', '')[:10]
            league = h.get('league', '')
            header = f"[{i}] {h.get('title', '')} ({date})"
            if league and league != 'other':
                header += f" — {league}"
            parts.append(f"{header}\n{h.get('body_clean', h.get('text', ''))}")
        return "\n\n---\n\n".join(parts)


# ─── Smoke-test (BM25 only, no GPU needed) ───────────────────────────────────

def build_bm25_only():
    """Verify BM25Okapi locally without loading FAISS or the embedding model."""
    log.info("📥 Loading chunks for BM25 build...")

    from preprocessing.pipeline import CHUNKS_FILE
    chunks = []
    with open(CHUNKS_FILE, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    texts = [c['text'] for c in chunks]
    log.info(f"Building BM25Okapi on {len(texts)} chunks...")
    tokenized_corpus = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    test_queries = [
        "أهداف الأهلي في الدوري",
        "تشكيل الزمالك أمام الاتحاد",
        "صلاح يسجل هدفاً",
    ]
    for q in test_queries:
        query_tokens = _tokenize(_normalize_ar(q))
        scores       = bm25.get_scores(query_tokens)
        top_indices  = np.argsort(scores)[::-1][:3]
        log.info(f"\nQuery: {q}")
        for i in top_indices:
            log.info(f"  [{scores[i]:.2f}] {chunks[i]['title'][:60]}")

    log.info("\n✅ BM25Okapi working correctly")
    return bm25


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_bm25_only()