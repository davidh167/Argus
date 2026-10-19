"""Does libargus's llama.cpp embedder agree with Ollama, and how fast is it?
Same GGUF weights (Ollama's nomic-embed-text blob) through two runtimes. Samples real
skill phrases (research/cluster-quality/data/phrases.json) and job-description excerpts
(research/ranking-eval/data/snapshot.jsonl), reports cosine agreement and throughput.
Run (Ollama must be running):
  PYTHONPATH=. poetry run python native/bench/compare_with_ollama.py"""

import ctypes
import json
import os
import random
import time
from pathlib import Path

import httpx
import numpy as np

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "build-release" / "libargus.dylib"
MODEL = os.path.expanduser(
    "~/.ollama/models/blobs/sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6"
)
EXP = HERE.parents[1]

lib = ctypes.CDLL(str(LIB))
lib.argus_embedder_open.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]
lib.argus_embedder_dim.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
lib.argus_embed_batch.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint8),
]
lib.argus_embedder_close.argtypes = [ctypes.c_void_p]

e = ctypes.c_void_p()
assert lib.argus_embedder_open(MODEL.encode(), 0, ctypes.byref(e)) == 0
dim = ctypes.c_size_t()
lib.argus_embedder_dim(e, ctypes.byref(dim))
dim = dim.value


def argus_embed(texts, chunk=250):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for s in range(0, len(texts), chunk):
        part = [t.encode() for t in texts[s : s + chunk]]
        arr = (ctypes.c_char_p * len(part))(*part)
        lens = (ctypes.c_size_t * len(part))(*[len(p) for p in part])
        buf = (ctypes.c_float * (len(part) * dim))()
        assert lib.argus_embed_batch(e, arr, lens, len(part), buf, dim, None) == 0
        out[s : s + len(part)] = np.frombuffer(buf, dtype=np.float32).reshape(len(part), dim)
    return out


def ollama_embed(texts, chunk=250):
    vecs = []
    for s in range(0, len(texts), chunk):
        r = httpx.post(
            "http://127.0.0.1:11434/api/embed",
            json={"model": "nomic-embed-text", "input": texts[s : s + chunk]},
            timeout=300,
        )
        vecs += r.json()["embeddings"]
    v = np.array(vecs, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


random.seed(0)
rows = json.loads((EXP / "research/cluster-quality/data/phrases.json").read_text())
phrases = sorted({s for r in rows for s in r["skills"]})
phrases = random.sample(phrases, 300)
snap = [
    json.loads(l)
    for l in (EXP / "research/ranking-eval/data/snapshot.jsonl").read_text().splitlines()
]
docs = ["search_document: " + f"{r['title']}\n{r['text'][:1200]}" for r in random.sample(snap, 100)]

lines = []
for name, texts in (
    ("300 skill phrases (bare, as the app embeds them)", phrases),
    ("100 job-description excerpts (search_document: prefix, 1,200 chars)", docs),
):
    ollama_embed(texts[:5])
    argus_embed(texts[:5])  # warm up both
    t = time.perf_counter()
    a = argus_embed(texts)
    t_argus = time.perf_counter() - t
    t = time.perf_counter()
    b = ollama_embed(texts)
    t_oll = time.perf_counter() - t
    cos = np.sum(a * b, axis=1)
    lines += [
        f"{name}:",
        f"  cosine libargus vs Ollama: min {cos.min():.6f}  mean {cos.mean():.6f}  p1 {np.percentile(cos, 1):.6f}",
        f"  throughput: libargus {len(texts) / t_argus:.0f} texts/s ({t_argus:.2f} s) | Ollama {len(texts) / t_oll:.0f} texts/s ({t_oll:.2f} s)",
    ]
(HERE.parent / "reports").mkdir(exist_ok=True)
(HERE.parent / "reports" / "ollama_agreement.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
lib.argus_embedder_close(e)
