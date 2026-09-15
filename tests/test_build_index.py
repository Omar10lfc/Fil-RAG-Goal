"""Unit tests for preprocessing/build_index.py.

Uses fake FAISS indexes + fake embedding models in tmp dirs — no 400MB
model load, no real index writes.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from preprocessing import build_index as bi


class FakeIndex:
    def __init__(self, dim: int = 4):
        self.d = dim
        self.vecs = np.zeros((0, dim), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self.vecs.shape[0])

    def add(self, embs: np.ndarray) -> None:
        self.vecs = np.vstack([self.vecs, np.asarray(embs, dtype=np.float32)])

    def search(self, q: np.ndarray, k: int = 1):
        # Cosine-ish: dot product against stored vecs; return best.
        scores = self.vecs @ q[0]
        order = np.argsort(scores)[::-1][:k]
        D = scores[order].reshape(1, -1)
        I = order.reshape(1, -1)
        return D.astype(np.float32), I.astype(np.int64)

    def reconstruct_n(self, i: int, n: int, out: np.ndarray) -> None:
        out[:] = self.vecs[i:i + n]


class FakeModel:
    def __init__(self, dim: int = 4):
        self.dim = dim

    def encode(self, texts, **kwargs):
        rng = np.random.default_rng(abs(hash(texts[0])) % (2 ** 32))
        vecs = rng.random((len(texts), self.dim), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        return vecs


@pytest.fixture()
def isolated_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(bi, "FAISS_DIR", tmp_path / "faiss_index")
    monkeypatch.setattr(bi, "INDEX_FILE", tmp_path / "faiss_index" / "index.bin")
    monkeypatch.setattr(bi, "META_FILE", tmp_path / "faiss_index" / "metadata.jsonl")
    monkeypatch.setattr(bi, "CONFIG_FILE", tmp_path / "faiss_index" / "config.json")
    monkeypatch.setattr(bi, "CHUNKS_FILE", tmp_path / "chunks.jsonl")
    return tmp_path


def _chunk(cid: str, date: str = "2026-03-01") -> dict:
    return {
        "chunk_id": cid, "article_id": cid.split("_")[0], "chunk_index": 0,
        "total_chunks": 1, "title": f"T-{cid}", "section": "",
        "article_type": "article", "pub_date": date, "teams": [],
        "league": "", "source_url": "", "text": f"text of {cid}",
        "body_clean": f"clean {cid}",
    }


def test_diff_new_returns_only_unseen(isolated_dirs):
    meta = [_chunk("a_0"), _chunk("b_0")]
    chunks = [_chunk("a_0"), _chunk("b_0"), _chunk("c_0")]
    assert [c["chunk_id"] for c in bi.diff_new(chunks, meta)] == ["c_0"]


def test_load_chunks_dedups_preserving_order(isolated_dirs, tmp_path):
    rows = [_chunk("a_0"), _chunk("b_0"), _chunk("a_0")]
    (tmp_path / "chunks.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    got = bi.load_chunks()
    assert [c["chunk_id"] for c in got] == ["a_0", "b_0"]


def test_check_counts_mismatch(isolated_dirs):
    idx = FakeIndex()
    idx.add(np.eye(4, dtype=np.float32)[:2])
    assert bi.check_counts(idx, [_chunk("a_0")], {"n_vectors": 2, "n_chunks": 1}) is not None
    assert bi.check_counts(idx, [_chunk("a_0"), _chunk("b_0")],
                           {"n_vectors": 2, "n_chunks": 2}) is None


def test_smoke_test_with_fake_index():
    idx = FakeIndex()
    idx.add(np.ones((3, 4), dtype=np.float32))
    bi.smoke_test(idx, FakeModel())  # must not raise


def test_prune_keeps_new_drops_old(monkeypatch, isolated_dirs, tmp_path):
    import faiss

    # Build a tiny real FlatIP index so prune() exercises reconstruct_n for real.
    dim = 4
    real = faiss.IndexFlatIP(dim)
    rng = np.random.default_rng(0)
    vecs = rng.random((3, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    real.add(vecs)

    meta = [_chunk("old_0", "2026-01-01"), _chunk("new_0", "2026-05-01"),
            _chunk("nodate_0", "")]
    (tmp_path / "faiss_index").mkdir()
    faiss.write_index(real, str(tmp_path / "faiss_index" / "index.bin"))
    (tmp_path / "faiss_index" / "metadata.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in meta), encoding="utf-8")
    (tmp_path / "faiss_index" / "config.json").write_text(
        json.dumps({"model_name": "m", "dim": dim, "n_vectors": 3, "n_chunks": 3}),
        encoding="utf-8")

    kept, cfg = bi.prune("2026-03-01")
    assert {m["chunk_id"] for m in kept} == {"new_0", "nodate_0"}
    assert cfg["n_vectors"] == 2 and cfg["n_chunks"] == 2
    # Reloaded index must still search.
    idx2 = faiss.read_index(str(tmp_path / "faiss_index" / "index.bin"))
    assert idx2.ntotal == 2
