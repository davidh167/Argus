"""Local dense embeddings via Ollama, for semantic similarity between
short skill/language phrases. app/services/roadmap.py's market_demand
clusters Stage 2's stage2_key_skills (2-4 word phrases like "distributed
systems design" or "PCB layout") -- TF-IDF cosine similarity chains
unrelated phrases together whenever they merely share one generic word
("design", "management", "operations"), a failure confirmed on a real corpus
("RF hardware design" and "Computer Vision" ended up in the same cluster). Dense embeddings don't
have that failure mode the same way, because similarity reflects meaning,
not shared tokens.

Requires a local Ollama instance (`brew services start ollama`, model
pulled via `ollama pull nomic-embed-text`) -- raises loudly (httpx's own
connection error) if unreachable rather than silently falling back to a
different representation, since this is a fixed piece of local
infrastructure on this machine, not an optional/flaky dependency.

Confirmed live against this install: a single /api/embed call with ~500+
inputs reliably crashes Ollama's internal tokenizer subprocess (400,
"connect: connection reset by peer" / "EOF"); 400 succeeds reliably, so
_EMBED_BATCH_SIZE=250 keeps real margin below that. Separately, and even
within that margin, the same crash has been observed to happen
intermittently (once in several dozen calls) -- but the service
self-recovers within seconds without a restart (confirmed via its own
logs: the very next request after a 400 succeeded normally ~10s later),
so a short retry is the correct fix, not a bigger margin or a restart."""

import time

import httpx
import numpy as np

from app.config import settings

OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"
_EMBED_BATCH_SIZE = 250
_EMBED_RETRY_DELAYS = (2, 5, 10)  # seconds; observed recovery is within ~10s

# Keyed on exact phrase text -- an embedding for a given string never
# changes, unlike TF-IDF's idf weights which depend on the whole corpus
# being clustered. Real key_skills phrases repeat heavily (about 7.5k
# distinct out of 16k instances in one corpus), so this saves most
# embedding calls across repeated market_demand requests for the life of
# the process.
_embedding_cache: dict[str, np.ndarray] = {}


def embed_texts(texts: list[str]) -> np.ndarray:
    """Returns an (len(texts), dim) array of L2-normalized embeddings, in
    input order, so cosine similarity between any two rows is just their
    dot product. Duplicate texts are embedded once each, not once per
    occurrence."""
    uncached = sorted({t for t in texts if t not in _embedding_cache})
    for start in range(0, len(uncached), _EMBED_BATCH_SIZE):
        batch = uncached[start : start + _EMBED_BATCH_SIZE]
        embeddings_out = _embed_batch(batch)
        for text, vec in zip(batch, embeddings_out):
            arr = np.array(vec, dtype=np.float64)
            norm = np.linalg.norm(arr)
            _embedding_cache[text] = arr / norm if norm > 0 else arr
    return np.stack([_embedding_cache[t] for t in texts])


def _embed_batch(batch: list[str]):
    """Dispatches on settings.embeddings_backend (see app/config.py)."""
    if settings.embeddings_backend == "native":
        from app.services import native_embedder

        return native_embedder.embed(batch)
    return _embed_batch_with_retry(batch)


def _embed_batch_with_retry(batch: list[str]) -> list[list[float]]:
    delays = (0, *_EMBED_RETRY_DELAYS)
    for attempt, delay in enumerate(delays):
        if delay:
            time.sleep(delay)
        resp = httpx.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBEDDING_MODEL, "input": batch},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()["embeddings"]
        if attempt == len(delays) - 1:
            resp.raise_for_status()
    # raise_for_status() only raises on 4xx/5xx, so a non-200 2xx/3xx on the last attempt lands here.
    raise RuntimeError(f"Ollama embeddings request ended with HTTP {resp.status_code}")
