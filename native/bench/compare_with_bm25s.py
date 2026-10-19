"""Real-corpus check of libargus's FTS5 BM25 against the shipped bm25s ranking.
Builds the index from research/ranking-eval/data/snapshot.jsonl, scores the
profile query, and writes data/extra_scores.npz so the ranking harness can evaluate
it alongside the other systems (python .../ranking-eval/evaluate.py).
Run (from the repo root, after `cmake --build native/build-release`):
  PYTHONPATH=. poetry run python native/bench/compare_with_bm25s.py"""

import ctypes
import json
import time
from pathlib import Path

import numpy as np
from bm25s.stopwords import STOPWORDS_EN

from app.services import relevance

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "build-release" / "libargus.dylib"
DATA = HERE.parents[1] / "research" / "ranking-eval" / "data"


class Hit(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int64), ("score", ctypes.c_double)]


lib = ctypes.CDLL(str(LIB))
lib.argus_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
lib.argus_upsert.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_char_p, ctypes.c_size_t]
lib.argus_search_bm25.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(Hit),
    ctypes.POINTER(ctypes.c_size_t),
]
lib.argus_close.argtypes = [ctypes.c_void_p]

rows = [json.loads(l) for l in (DATA / "snapshot.jsonl").read_text().splitlines()]
query = json.loads((DATA / "meta.json").read_text())["profile_query"]
docs = [f"{r['title']} {r['text']}" for r in rows]

h = ctypes.c_void_p()
assert lib.argus_open(b":memory:", ctypes.byref(h)) == 0
t = time.perf_counter()
for i, d in enumerate(docs):
    b = d.encode()
    assert lib.argus_upsert(h, i, b, len(b)) == 0
build_s = time.perf_counter() - t


def scores_for(q: str) -> np.ndarray:
    b = q.encode()
    hits = (Hit * len(docs))()
    n = ctypes.c_size_t()
    t0 = time.perf_counter()
    assert lib.argus_search_bm25(h, b, len(b), len(docs), hits, ctypes.byref(n)) == 0
    dt = time.perf_counter() - t0
    out = np.zeros(len(docs))
    for k in range(n.value):
        out[hits[k].id] = hits[k].score
    return out, dt


no_stop = " ".join(w for w in query.split() if w.lower().strip(",.()") not in STOPWORDS_EN)
raw, t_raw = scores_for(query)
filtered, t_filtered = scores_for(no_stop)
ref = np.array(relevance.bm25_scores(docs, query))

top = lambda s, k: set(np.argsort(-s)[:k])
print(f"index build over {len(docs)} docs: {build_s * 1000:.0f} ms")
for name, s, dt in (
    ("raw query", raw, t_raw),
    ("stopwords removed from query", filtered, t_filtered),
):
    print(
        f"{name}: query {dt * 1000:.1f} ms | top-10 overlap with bm25s {len(top(s, 10) & top(ref, 10))}/10 | "
        f"top-50 {len(top(s, 50) & top(ref, 50))}/50 | matched docs {int((s > 0).sum())} (bm25s: {int((ref > 0).sum())})"
    )
np.savez(DATA / "extra_scores.npz", fts5_raw=raw, fts5_nostop=filtered)
lib.argus_close(h)
