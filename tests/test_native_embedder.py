import os
from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.services import embeddings, native_embedder

OLLAMA_BLOB = (
    Path.home()
    / ".ollama/models/blobs/sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6"
)
MODEL = Path(os.environ["ARGUS_MODEL_PATH"]) if os.environ.get("ARGUS_MODEL_PATH") else OLLAMA_BLOB

needs_native = pytest.mark.skipif(
    not (native_embedder.DEFAULT_LIB.exists() and MODEL.exists()),
    reason="needs the built libargus and a GGUF embedding model (set ARGUS_MODEL_PATH)",
)


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(settings, "embeddings_backend", "native")
    monkeypatch.setattr(settings, "argus_model_path", str(MODEL))
    monkeypatch.setattr(settings, "argus_lib_path", None)
    monkeypatch.setattr(embeddings, "_embedding_cache", {})
    yield
    native_embedder.close()  # Metal asserts if a model is loaded at process exit; also keeps tests isolated


@needs_native
def test_native_vectors_are_unit_length_with_the_models_dimension(native):
    v = embeddings.embed_texts(["PyTorch", "distributed systems design"])
    assert v.shape == (2, 768)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


@needs_native
def test_native_embeddings_capture_meaning_and_are_cached(native):
    v = embeddings.embed_texts(["machine learning", "deep learning", "pastry baking"])
    assert v[0] @ v[1] > v[0] @ v[2]
    # A second call is served from the per-phrase cache without touching the model again.
    native_embedder.close()
    again = embeddings.embed_texts(["machine learning", "deep learning", "pastry baking"])
    assert np.array_equal(v, again)


def test_native_backend_without_a_model_fails_loudly(monkeypatch):
    monkeypatch.setattr(settings, "embeddings_backend", "native")
    monkeypatch.setattr(settings, "argus_model_path", None)
    monkeypatch.setattr(embeddings, "_embedding_cache", {})
    with pytest.raises(RuntimeError, match="ARGUS_MODEL_PATH|libargus not found"):
        embeddings.embed_texts(["anything"])


def test_native_backend_with_a_missing_library_fails_loudly(monkeypatch):
    monkeypatch.setattr(settings, "embeddings_backend", "native")
    monkeypatch.setattr(settings, "argus_lib_path", "/no/such/libargus.dylib")
    monkeypatch.setattr(embeddings, "_embedding_cache", {})
    with pytest.raises(RuntimeError, match="libargus not found"):
        embeddings.embed_texts(["anything"])


def test_default_backend_is_ollama():
    assert settings.embeddings_backend == "ollama"
