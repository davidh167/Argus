"""Dense embeddings for short skill/language phrases. app/services/roadmap.py's market_demand
clusters Stage 2's stage2_key_skills (2-4 word phrases like "distributed
systems design" or "PCB layout") -- TF-IDF cosine similarity chains
unrelated phrases together whenever they merely share one generic word
("design", "management", "operations"), a failure confirmed on a real corpus
("RF hardware design" and "Computer Vision" ended up in the same cluster). Dense embeddings don't
have that failure mode the same way, because similarity reflects meaning,
not shared tokens.

Two backends, chosen explicitly by EMBEDDINGS_BACKEND (no auto-detection, no fallback between
them -- which runtime produced a vector is something the caller should know):

- "native" (default): the GGUF model runs in-process through libargus (app/native). Needs the built
  library and the model file; raises with the exact build / fetch command if either is missing.
  Vectors are also persisted in an index file (app/services/vector_cache.py), so a restart does not
  re-embed phrases it has seen. Lookup order: in-process dict, then the index file, then the model.
- "ollama": a local Ollama instance (`brew services start ollama`, model pulled via
  `ollama pull nomic-embed-text`); raises loudly (httpx's own connection error) if unreachable.
  Nothing is persisted for it.

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
from numpy.typing import ArrayLike

from app import native
from app.config import settings
from app.services.vector_cache import VectorCache

OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"
_EMBED_BATCH_SIZE = 250
_EMBED_RETRY_DELAYS = (2, 5, 10)  # seconds; observed recovery is within ~10s

# Keyed on exact phrase text -- an embedding for a given string never
# changes, unlike TF-IDF's idf weights which depend on the whole corpus
# being clustered. Real key_skills phrases repeat heavily (about 7.5k
# distinct out of 16k instances in one corpus), so this saves most
# embedding calls across repeated market_demand requests for the life of
# the process. Values are float64 views of the float32 vectors the persistent cache stores, so a
# vector read back from disk is identical to the one that was just embedded.
_embedding_cache: dict[str, np.ndarray] = {}
_persistent: VectorCache | None = None


def embed_texts(texts: list[str]) -> np.ndarray:
    """Returns an (len(texts), dim) array of L2-normalized embeddings, in
    input order, so cosine similarity between any two rows is just their
    dot product. Duplicate texts are embedded once each, not once per
    occurrence."""
    uncached = sorted({t for t in texts if t not in _embedding_cache})
    persistent = (
        _persistent_cache() if uncached and settings.embeddings_backend == "native" else None
    )
    if persistent is not None:
        for text, vec in persistent.lookup(uncached).items():
            _embedding_cache[text] = vec.astype(np.float64)
        uncached = [t for t in uncached if t not in _embedding_cache]
    for start in range(0, len(uncached), _EMBED_BATCH_SIZE):
        batch = uncached[start : start + _EMBED_BATCH_SIZE]
        embeddings_out = _embed_batch(batch)
        vectors = {text: _normalized_float32(vec) for text, vec in zip(batch, embeddings_out)}
        if persistent is not None:
            persistent.save(vectors)
        for text, vec32 in vectors.items():
            _embedding_cache[text] = vec32.astype(np.float64)
    return np.stack([_embedding_cache[t] for t in texts])


def _normalized_float32(vec: ArrayLike) -> np.ndarray:
    """Normalizes in float64, then rounds to float32 -- the precision both stored and fresh vectors share."""
    arr = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(arr)
    return (arr / norm if norm > 0 else arr).astype(np.float32)


def _persistent_cache() -> VectorCache:
    """The process-wide persistent cache, opened (and the model identity verified) on first use."""
    global _persistent
    if _persistent is None:
        _persistent = _open_persistent()
    return _persistent


def _open_persistent() -> VectorCache:
    return VectorCache(native.vector_index(), native.model_path())


def close() -> None:
    """Releases the native model and index, and forgets the persistent cache that pointed at them."""
    global _persistent
    _persistent = None
    native.close()


def _embed_batch(batch: list[str]):
    """Dispatches on settings.embeddings_backend (see app/config.py)."""
    if settings.embeddings_backend == "native":
        return native.embedder().embed(batch)
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
