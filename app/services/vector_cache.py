"""Persistent phrase -> embedding cache, so a process restart does not re-embed known phrases.

Entries live in a libargus index file (app/native), keyed by a 64-bit hash of the text. The C API
stores only (id, vector), so the original text is stored next to it and compared on every lookup:
a hash collision is detected and handled explicitly (the colliding phrase is simply not cached),
never answered with another phrase's vector.

The file is stamped with the SHA-256 of the embedding model that produced its vectors. Opening it
with a different model raises ModelMismatchError rather than mixing vectors from two models.
"""

import hashlib
import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from app.native import ModelMismatchError

logger = logging.getLogger(__name__)

_MODEL_SHA_KEY = "model_sha256"
_MODEL_STAT_KEY = "model_stat"


class VectorStore(Protocol):
    """What the cache needs from storage; app.native.Index implements it."""

    path: str

    def get_text(self, key: int) -> str | None: ...
    def get_vector(self, key: int) -> np.ndarray | None: ...
    def put_records(self, records: list[tuple[int, str, np.ndarray]]) -> None: ...
    def meta_get(self, name: str) -> str | None: ...
    def meta_set(self, name: str, value: str) -> None: ...


def text_key(text: str) -> int:
    """Stable signed 64-bit id for a text: the first 8 bytes of its BLAKE2b digest."""
    return int.from_bytes(
        hashlib.blake2b(text.encode(), digest_size=8).digest(), "big", signed=True
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verified_model_sha256(store: VectorStore, model_file: Path) -> str:
    """The model file's SHA-256, hashing it only when its size or mtime differ from what the index
    recorded the last time it was verified (hashing 274 MB on every start would not be cheap).
    Raises ModelMismatchError if the index was stamped with a different model."""
    st = model_file.stat()
    stat = f"{st.st_size}:{st.st_mtime_ns}"
    stamped = store.meta_get(_MODEL_SHA_KEY)
    if stamped is not None and store.meta_get(_MODEL_STAT_KEY) == stat:
        return stamped
    sha = _sha256_file(model_file)
    if stamped is not None and stamped != sha:
        raise ModelMismatchError(
            f"The embedding index {store.path} was built with a different model "
            f"(sha256 {stamped[:12]}...) than the current one ({model_file}, sha256 {sha[:12]}...). "
            f"Vectors from different models must not be mixed. Delete the index to rebuild it "
            f"from scratch: rm {store.path}*"
        )
    store.meta_set(_MODEL_SHA_KEY, sha)
    store.meta_set(_MODEL_STAT_KEY, stat)
    return sha


class VectorCache:
    def __init__(self, store: VectorStore, model_file: Path) -> None:
        _verified_model_sha256(store, model_file)
        self._store = store

    def lookup(self, texts: list[str]) -> dict[str, np.ndarray]:
        """Float32 vectors for the texts that are cached, exactly as they were saved."""
        found: dict[str, np.ndarray] = {}
        for text in texts:
            key = text_key(text)
            stored = self._store.get_text(key)
            if stored is None:
                continue
            if stored != text:
                logger.warning("embedding cache hash collision between %r and %r", text, stored)
                continue
            vec = self._store.get_vector(key)
            if vec is not None:
                found[text] = vec
        return found

    def save(self, items: dict[str, np.ndarray]) -> None:
        """Stores float32 vectors in one transaction. A text whose hash is already taken by a
        different text is not stored (it will be re-embedded each process); the existing entry is
        left alone."""
        records = []
        batch_texts: dict[int, str] = {}
        for text, vector in items.items():
            key = text_key(text)
            stored = batch_texts.get(key) or self._store.get_text(key)
            if stored is not None and stored != text:
                logger.warning("embedding cache hash collision between %r and %r", text, stored)
                continue
            batch_texts[key] = text
            records.append((key, text, vector))
        if records:
            self._store.put_records(records)
