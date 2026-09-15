"""
FilGoalBot — FAISS index builder (incremental + rebuild + prune + smoke test)
===============================================================================
Missing link between preprocessing/pipeline.py (writes data/processed/chunks.jsonl)
and retrieval/hybrid_retriever.py (reads faiss_index/index.bin + metadata.jsonl).

The daily refresh workflow runs:
    python -m scraper.filgoal_scraper --newest-only
    python -m preprocessing.pipeline
    python -m preprocessing.build_index            # incremental (default)
    python -m preprocessing.build_index --smoke-only

CLI:
    python -m preprocessing.build_index                # incremental append
    python -m preprocessing.build_index --rebuild      # re-embed everything
    python -m preprocessing.build_index --prune-older-than YYYY-MM-DD
    python -m preprocessing.build_index --smoke-only   # CI gate, no writes

Conventions (must stay in sync with retrieval/hybrid_retriever.py):
    - e5 passage prefix "passage: " for documents ("query: " is for queries).
    - L2-normalized embeddings + IndexFlatIP (= cosine via dot product).
    - model name / dim / metric are read from faiss_index/config.json,
      never hard-coded here.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger("build_index")

FAISS_DIR = Path("faiss_index")
INDEX_FILE = FAISS_DIR / "index.bin"
META_FILE = FAISS_DIR / "metadata.jsonl"
CONFIG_FILE = FAISS_DIR / "config.json"
CHUNKS_FILE = Path("data/processed/chunks.jsonl")

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
EMBED_BATCH_SIZE = 64

# Metadata fields the retriever + RAG pipeline read. Kept explicit so a
# pipeline schema change fails loudly here instead of producing a silent
# index/metadata skew.
META_FIELDS = (
    "chunk_id", "article_id", "chunk_index", "total_chunks", "title",
    "section", "article_type", "pub_date", "teams", "league",
    "source_url", "text", "body_clean",
)


# ─── Load helpers ─────────────────────────────────────────────────────────────

def load_existing() -> tuple[object | None, list[dict], dict]:
    """Return (faiss_index|None, metadata_rows, config).

    Config (model/dim/metric) comes from faiss_index/config.json and is never
    hard-coded. Missing files → (None, [], {}) so the first build works from
    an empty directory.
    """
    metadata: list[dict] = []
    config: dict = {}
    index = None

    if CONFIG_FILE.is_file():
        try:
            config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Could not read {CONFIG_FILE}: {e} — treating as empty")
            config = {}

    if META_FILE.is_file():
        with open(META_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        metadata.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if INDEX_FILE.is_file():
        import faiss

        try:
            index = faiss.read_index(str(INDEX_FILE))
        except Exception as e:
            log.warning(f"Could not read {INDEX_FILE}: {e} — treating as missing")
            index = None

    return index, metadata, config


def load_chunks() -> list[dict]:
    """Load chunks.jsonl, deduplicated by chunk_id, preserving file order."""
    chunks: list[dict] = []
    seen: set[str] = set()
    if not CHUNKS_FILE.is_file():
        return chunks
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = c.get("chunk_id", "")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            chunks.append(c)
    return chunks


def diff_new(chunks: list[dict], metadata: list[dict]) -> list[dict]:
    """Return chunks whose chunk_id is not yet in metadata.jsonl."""
    known = {m.get("chunk_id", "") for m in metadata}
    return [c for c in chunks if c.get("chunk_id", "") not in known]


def _meta_row(chunk: dict) -> dict:
    """Project a pipeline chunk onto the retriever metadata schema."""
    row = {k: chunk.get(k, "" if k not in ("chunk_index", "total_chunks", "teams") else 0)
           for k in META_FIELDS}
    # Normalise non-string defaults the dict-comprehension above can't express.
    if not isinstance(row.get("teams"), list):
        row["teams"] = chunk.get("teams", []) or []
    if not isinstance(row.get("chunk_index"), int):
        row["chunk_index"] = chunk.get("chunk_index", 0) or 0
    if not isinstance(row.get("total_chunks"), int):
        row["total_chunks"] = chunk.get("total_chunks", 1) or 1
    # text is the normalised embedding string; body_clean is display text.
    # Fall back so the retriever never gets an empty embedding input.
    if not row.get("text"):
        row["text"] = chunk.get("body_clean", "") or ""
    return row


# ─── Embedding ────────────────────────────────────────────────────────────────

def embed_passages(model: object, texts: list[str]) -> np.ndarray:
    """Embed document texts with the e5 "passage: " prefix, L2-normalized.

    Returns float32 (N, dim) ready for IndexFlatIP. The retriever encodes
    queries with the "query: " prefix — the asymmetry is intentional per the
    multilingual-e5 spec.
    """
    prefixed = ["passage: " + t for t in texts]
    embs = model.encode(  # type: ignore[attr-defined]
        prefixed,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embs, dtype=np.float32)


def _load_model(model_name: str) -> object:
    from sentence_transformers import SentenceTransformer

    log.info(f"Loading sentence model: {model_name}")
    return SentenceTransformer(model_name)


def _new_index(dim: int) -> object:
    import faiss

    return faiss.IndexFlatIP(dim)


# ─── Writes ───────────────────────────────────────────────────────────────────

def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_and_save(
    index: object | None,
    metadata: list[dict],
    config: dict,
    new_chunks: list[dict],
    model_name: str,
) -> tuple[object, list[dict], dict]:
    """Embed new_chunks, append to index + metadata, persist all three artifacts.

    Updates config n_vectors/n_chunks in place. Creates the index on first run.
    """
    import faiss

    model = _load_model(model_name)
    texts = [c.get("text", "") or c.get("body_clean", "") for c in new_chunks]
    embs = embed_passages(model, texts)
    dim = int(embs.shape[1])

    if index is None:
        index = _new_index(dim)
    # Guard against dim drift between the committed index and the model.
    if getattr(index, "d", dim) != dim:
        raise SystemExit(
            f"Embedding dim {dim} != index dim {index.d} "
            f"(model={model_name}). Re-run with --rebuild."
        )

    index.add(embs)  # type: ignore[attr-defined]
    metadata.extend(_meta_row(c) for c in new_chunks)

    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    # Atomic index write: faiss can't write to our tmp name via replace on all
    # platforms cleanly, so write tmp then os-replace.
    tmp_index = FAISS_DIR / "index.bin.tmp"
    faiss.write_index(index, str(tmp_index))  # type: ignore[attr-defined]
    tmp_index.replace(INDEX_FILE)
    _atomic_write_jsonl(META_FILE, metadata)
    config = {
        "model_name": model_name,
        "dim": dim,
        "metric": config.get("metric", "ip"),
        "n_vectors": int(index.ntotal),  # type: ignore[attr-defined]
        "n_chunks": len(metadata),
    }
    _atomic_write_json(CONFIG_FILE, config)
    log.info(f"Appended {len(new_chunks)} chunks — ntotal={index.ntotal}")  # type: ignore[attr-defined]
    return index, metadata, config


def rebuild_from_chunks(chunks: list[dict], model_name: str) -> tuple[object, list[dict], dict]:
    """Re-embed every chunk from scratch. Used by --rebuild (e.g. model swap)."""
    import faiss

    if not chunks:
        raise SystemExit(f"No chunks to rebuild from ({CHUNKS_FILE} missing or empty).")
    model = _load_model(model_name)
    texts = [c.get("text", "") or c.get("body_clean", "") for c in chunks]
    log.info(f"Encoding {len(texts)} chunks with {model_name}...")
    embs = embed_passages(model, texts)
    dim = int(embs.shape[1])

    index = _new_index(dim)
    index.add(embs)  # type: ignore[attr-defined]
    metadata = [_meta_row(c) for c in chunks]

    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_index = FAISS_DIR / "index.bin.tmp"
    faiss.write_index(index, str(tmp_index))  # type: ignore[attr-defined]
    tmp_index.replace(INDEX_FILE)
    _atomic_write_jsonl(META_FILE, metadata)
    config = {
        "model_name": model_name,
        "dim": dim,
        "metric": "ip",
        "n_vectors": int(index.ntotal),  # type: ignore[attr-defined]
        "n_chunks": len(metadata),
    }
    _atomic_write_json(CONFIG_FILE, config)
    log.info(f"Rebuilt index — ntotal={index.ntotal}")  # type: ignore[attr-defined]
    return index, metadata, config


def prune(cutoff_iso: str) -> tuple[list[dict], dict]:
    """Drop metadata rows older than YYYY-MM-DD and subset the index WITHOUT
    re-embedding, via index.reconstruct_n. Rows with missing/unparseable dates
    are kept (better to keep than to silently drop evidence)."""
    import faiss

    try:
        cutoff = datetime.fromisoformat(cutoff_iso).date().isoformat()
    except ValueError:
        raise SystemExit(f"--prune-older-than expects YYYY-MM-DD, got {cutoff_iso!r}")

    index, metadata, config = load_existing()
    if index is None or not metadata:
        raise SystemExit("Nothing to prune — index or metadata missing.")

    keep_idx = [i for i, m in enumerate(metadata)
                if not str(m.get("pub_date", ""))[:10] or str(m.get("pub_date", ""))[:10] >= cutoff]
    if len(keep_idx) == len(metadata):
        log.info("Prune: nothing older than cutoff — no-op.")
        return metadata, config

    try:
        ntotal = int(index.ntotal)  # type: ignore[attr-defined]
        all_vecs = np.empty((ntotal, int(index.d)), dtype=np.float32)  # type: ignore[attr-defined]
        index.reconstruct_n(0, ntotal, all_vecs)  # type: ignore[attr-defined]
    except Exception as e:
        raise SystemExit(
            f"Prune needs index.reconstruct_n support (Flat index). Got: {e}. "
            "Re-run with --rebuild instead."
        )

    kept_vecs = all_vecs[np.array(keep_idx, dtype=np.int64)]
    new_index = _new_index(int(kept_vecs.shape[1]))
    new_index.add(kept_vecs)  # type: ignore[attr-defined]
    metadata = [metadata[i] for i in keep_idx]

    tmp_index = FAISS_DIR / "index.bin.tmp"
    faiss.write_index(new_index, str(tmp_index))  # type: ignore[attr-defined]
    tmp_index.replace(INDEX_FILE)
    _atomic_write_jsonl(META_FILE, metadata)
    config["n_vectors"] = int(new_index.ntotal)  # type: ignore[attr-defined]
    config["n_chunks"] = len(metadata)
    _atomic_write_json(CONFIG_FILE, config)
    log.info(f"Pruned to {len(metadata)} chunks (cutoff {cutoff}).")
    return metadata, config


# ─── Consistency + smoke ──────────────────────────────────────────────────────

def check_counts(index: object | None, metadata: list[dict], config: dict) -> str | None:
    """Return an error string when counts disagree, else None."""
    n_meta = len(metadata)
    n_idx = int(index.ntotal) if index is not None else -1  # type: ignore[attr-defined]
    n_cfg = int(config.get("n_vectors", -1)) if config else -1
    if index is None:
        return "index.bin missing"
    if n_idx != n_meta or (config and n_cfg not in (-1, n_idx)):
        return (
            f"count mismatch: config.n_vectors={n_cfg} "
            f"index.ntotal={n_idx} len(metadata)={n_meta}"
        )
    if config and int(config.get("n_chunks", n_meta)) != n_meta:
        return f"config.n_chunks={config.get('n_chunks')} != len(metadata)={n_meta}"
    return None


def smoke_test(index: object | None = None, model: object | None = None) -> None:
    """Encode 'query: الأهلي', search top-1, assert a sane hit.

    Loads the committed index + config model when not passed explicitly
    (tests inject fakes). Raises SystemExit on failure so CI gates on it.
    """
    if index is None or model is None:
        index, _, config = load_existing()
        if index is None:
            raise SystemExit("smoke test: index.bin missing — build the index first.")
        model_name = str(config.get("model_name", DEFAULT_MODEL))
        model = _load_model(model_name)

    q_emb = model.encode(  # type: ignore[attr-defined]
        ["query: الأهلي"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    try:
        D, I = index.search(q_emb, k=1)  # type: ignore[attr-defined]
    except Exception as e:
        raise SystemExit(f"smoke test: index.search failed: {e}")
    if I.size == 0 or int(I[0][0]) < 0 or int(I[0][0]) >= int(index.ntotal):  # type: ignore[attr-defined]
        raise SystemExit(f"smoke test: invalid top-1 id {I!r}.")
    score = float(D[0][0])
    if not np.isfinite(score):
        raise SystemExit(f"smoke test: non-finite score {score!r}.")
    log.info(f"smoke test ok — top-1 id={int(I[0][0])} score={score:.4f}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FilGoalBot FAISS index builder")
    parser.add_argument("--rebuild", action="store_true",
                        help="Re-embed every chunk from scratch.")
    parser.add_argument("--prune-older-than", default=None, metavar="YYYY-MM-DD",
                        help="Drop rows older than this date (no re-embedding).")
    parser.add_argument("--smoke-only", action="store_true",
                        help="Only run count-consistency + smoke test, no writes.")
    parser.add_argument("--model", default=None,
                        help="Embedding model (default: config.json or e5-base).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.smoke_only:
        index, metadata, config = load_existing()
        err = check_counts(index, metadata, config)
        if err:
            log.error(f"❌ {err}")
            return 1
        try:
            smoke_test(index)
        except SystemExit as e:
            log.error(f"❌ {e}")
            return 1
        log.info(f"✅ counts consistent (n={len(metadata)}), smoke ok")
        return 0

    if args.prune_older_than:
        try:
            prune(args.prune_older_than)
        except SystemExit as e:
            log.error(f"❌ {e}")
            return 1
        index, metadata, config = load_existing()
        err = check_counts(index, metadata, config)
        if err:
            log.error(f"❌ {err}")
            return 1
        return 0

    chunks = load_chunks()
    if not chunks:
        log.error(f"❌ No chunks found at {CHUNKS_FILE} — run preprocessing.pipeline first.")
        return 1

    index, metadata, config = load_existing()
    model_name = args.model or str(config.get("model_name", DEFAULT_MODEL))

    if args.rebuild:
        try:
            index, metadata, config = rebuild_from_chunks(chunks, model_name)
        except SystemExit as e:
            log.error(f"❌ {e}")
            return 1
    else:
        new = diff_new(chunks, metadata)
        if not new:
            log.info(f"✅ Index up to date ({len(metadata)} chunks, 0 new).")
        else:
            log.info(f"Appending {len(new)} new chunks ({len(metadata)} existing)...")
            try:
                index, metadata, config = append_and_save(index, metadata, config, new, model_name)
            except SystemExit as e:
                log.error(f"❌ {e}")
                return 1

    err = check_counts(index, metadata, config)
    if err:
        log.error(f"❌ {err}")
        return 1
    try:
        smoke_test(index)
    except SystemExit as e:
        log.error(f"❌ {e}")
        return 1
    log.info("✅ build_index done — counts consistent, smoke ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
