"""Quality of libargus's dense and hybrid search on the ranking harness's corpus.
Embeds every posting with the in-process llama.cpp embedder, scores through the real index
(FTS5 BM25, usearch dense, RRF hybrid), and adds the systems to
research/ranking-eval/data/extra_scores.npz; then run ranking-eval/evaluate.py.
Query/document prefixes and the 1,200-char excerpt match the harness's own embedding scorer.
Run: PYTHONPATH=. poetry run python native/bench/eval_hybrid.py"""

import ctypes
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "build-release" / "libargus.dylib"
DATA = HERE.parents[1] / "research" / "ranking-eval" / "data"
MODEL = os.environ.get("ARGUS_MODEL") or os.path.expanduser(
    "~/.ollama/models/blobs/sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6"
)
DOC_CHARS = 1200
CAND = 100  # the library's hybrid candidate depth


class Hit(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int64), ("score", ctypes.c_double)]


lib = ctypes.CDLL(str(LIB))
vp, f32p = ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)
lib.argus_embedder_open.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(vp)]
lib.argus_embed_batch.argtypes = [
    vp,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_size_t,
    f32p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint8),
]
lib.argus_open_with_vectors.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(vp)]
lib.argus_upsert.argtypes = [vp, ctypes.c_int64, ctypes.c_char_p, ctypes.c_size_t]
lib.argus_upsert_vector.argtypes = [vp, ctypes.c_int64, f32p, ctypes.c_size_t]
lib.argus_search_bm25.argtypes = [
    vp,
    ctypes.c_char_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(Hit),
    ctypes.POINTER(ctypes.c_size_t),
]
lib.argus_search_dense.argtypes = [
    vp,
    f32p,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(Hit),
    ctypes.POINTER(ctypes.c_size_t),
]
lib.argus_search_hybrid.argtypes = [
    vp,
    ctypes.c_char_p,
    ctypes.c_size_t,
    f32p,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(Hit),
    ctypes.POINTER(ctypes.c_size_t),
]

emb = vp()
assert lib.argus_embedder_open(MODEL.encode(), 0, ctypes.byref(emb)) == 0
DIM = 768


def embed(texts, chunk=250):
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for s in range(0, len(texts), chunk):
        part = [t.encode() for t in texts[s : s + chunk]]
        arr = (ctypes.c_char_p * len(part))(*part)
        lens = (ctypes.c_size_t * len(part))(*[len(p) for p in part])
        buf = (ctypes.c_float * (len(part) * DIM))()
        assert lib.argus_embed_batch(emb, arr, lens, len(part), buf, DIM, None) == 0
        out[s : s + len(part)] = np.frombuffer(buf, dtype=np.float32).reshape(len(part), DIM)
    return out


rows = [json.loads(l) for l in (DATA / "snapshot.jsonl").read_text().splitlines()]
query = json.loads((DATA / "meta.json").read_text())["profile_query"]
n = len(rows)

t = time.perf_counter()
doc_vecs = embed([f"search_document: {r['title']}\n{r['text'][:DOC_CHARS]}" for r in rows])
q_vec = embed([f"search_query: {query}"])[0]
embed_s = time.perf_counter() - t

idx = vp()
assert lib.argus_open_with_vectors(b":memory:", DIM, ctypes.byref(idx)) == 0
t = time.perf_counter()
for i, r in enumerate(rows):
    b = f"{r['title']} {r['text']}".encode()
    assert lib.argus_upsert(idx, i, b, len(b)) == 0
    assert lib.argus_upsert_vector(idx, i, doc_vecs[i].ctypes.data_as(f32p), DIM) == 0
index_s = time.perf_counter() - t


def to_scores(fn, *args, k):
    hits = (Hit * k)()
    cnt = ctypes.c_size_t()
    t0 = time.perf_counter()
    assert fn(idx, *args, k, hits, ctypes.byref(cnt)) == 0
    dt = time.perf_counter() - t0
    s = np.zeros(n)
    for j in range(cnt.value):
        s[hits[j].id] = hits[j].score
    return s, dt, cnt.value


qb = query.encode()
qp = q_vec.ctypes.data_as(f32p)
bm25, t_b, _ = to_scores(lib.argus_search_bm25, qb, len(qb), k=n)
dense, t_d, _ = to_scores(lib.argus_search_dense, qp, DIM, k=n)
hybrid, t_h, n_h = to_scores(lib.argus_search_hybrid, qb, len(qb), qp, DIM, k=n)


def rrf_full(*score_vectors, k=60):
    fused = np.zeros(n)
    for s in score_vectors:
        order = np.argsort(-s, kind="stable")
        matched = s[order] > 0
        for rank, (i, ok) in enumerate(zip(order, matched), 1):
            if ok:
                fused[i] += 1.0 / (k + rank)
    return fused


full = rrf_full(bm25, dense)
top = lambda s, k: set(np.argsort(-s)[:k])
print(f"embedded {n} docs in {embed_s:.1f} s; indexed (FTS5 + vectors) in {index_s:.1f} s")
print(
    f"query latency: bm25 {t_b * 1000:.1f} ms | dense {t_d * 1000:.1f} ms | hybrid {t_h * 1000:.1f} ms ({n_h} docs scored)"
)
print(
    f"hybrid(top-100 lists) vs full-ranking RRF top-50 overlap: {len(top(hybrid, 50) & top(full, 50))}/50"
)

# --- Does a natural-language profile query help? The default query is a bare skills list. ---
from app.db import SessionLocal  # noqa: E402
from app.schemas.profile import ProfileSnapshotData  # noqa: E402
from app.services import profile as profile_service  # noqa: E402

with SessionLocal() as db:
    snap = ProfileSnapshotData.model_validate(profile_service.get_current_profile(db).snapshot)
narrative = " ".join(
    [f"{e.title}: " + "; ".join(e.signals) + "." for e in snap.experience]
    + [f"{p.name}: {p.description}" for p in snap.projects]
    + ["Skills: " + ", ".join(snap.skills.core + snap.skills.developing) + "."]
)
nb = narrative.encode()
nvec = embed([f"search_query: {narrative}"])[0]
nqp = nvec.ctypes.data_as(f32p)
bm25_n, _, _ = to_scores(lib.argus_search_bm25, nb, len(nb), k=n)
dense_n, _, _ = to_scores(lib.argus_search_dense, nqp, DIM, k=n)
full_n = rrf_full(bm25_n, dense_n)
print(f"narrative query: {len(narrative.split())} words")

path = DATA / "extra_scores.npz"
extra = dict(np.load(path)) if path.exists() else {}
extra.update(
    {
        "argus_dense": dense,
        "argus_hybrid100": hybrid,
        "argus_rrf_full": full,
        "narr_bm25": bm25_n,
        "narr_dense": dense_n,
        "narr_rrf_full": full_n,
    }
)
np.savez(path, **extra)

# llama.cpp's Metal backend asserts if the process exits while a model is still loaded.
lib.argus_close.argtypes = [vp]
lib.argus_embedder_close.argtypes = [vp]
lib.argus_close(idx)
lib.argus_embedder_close(emb)
