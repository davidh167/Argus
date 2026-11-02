"""A deterministic stand-in for app.services.embeddings.embed_texts, so market_demand can be timed
without Ollama or the native library.

Each word gets a fixed pseudo-random vector (seeded from a hash of the word) and a phrase is the
normalized sum of its words' vectors. Phrases that share words therefore land near each other, which
gives the clustering code a non-trivial similarity structure, and the same text always yields the same
vector in every process. The dimension matches nomic-embed-text (768) so the matmul and linkage costs
are the size they would be with the real model.

What this does NOT model: the real model's cost (the stub is a dictionary lookup and a sum) or its
similarity distribution. market_demand times measure the clustering code, not embedding.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import blake2b
from unittest.mock import patch

import numpy as np

DIM = 768

_word_vectors: dict[str, np.ndarray] = {}


def _word_vector(word: str) -> np.ndarray:
    vec = _word_vectors.get(word)
    if vec is None:
        seed = int.from_bytes(blake2b(word.encode(), digest_size=8).digest(), "big")
        vec = np.random.default_rng(seed).standard_normal(DIM)
        _word_vectors[word] = vec
    return vec


def stub_embed_texts(texts: list[str]) -> np.ndarray:
    """(len(texts), DIM) float64, rows L2-normalized, in input order."""
    out = np.empty((len(texts), DIM), dtype=np.float64)
    for i, text in enumerate(texts):
        total = sum((_word_vector(w) for w in text.casefold().split()), np.zeros(DIM))
        norm = np.linalg.norm(total)
        out[i] = total / norm if norm > 0 else total
    return out


@contextmanager
def stubbed_embedder() -> Iterator[None]:
    from app.services import embeddings

    with patch.object(embeddings, "embed_texts", stub_embed_texts):
        yield
