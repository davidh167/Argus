"""Native embeddings and the persistent vector cache.

The cache logic (collision handling, model identity, restart behaviour) is tested with a counting
stub in place of the model, against both an in-memory store (runs everywhere, including CI, which
has neither the library nor the model) and the real libargus index (skipped without the library).
Tests that run the real model skip without the library and a GGUF model file.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pytest

from app import native
from app.config import settings
from app.native import ModelMismatchError, NativeError
from app.services import embeddings, vector_cache
from app.services.vector_cache import VectorCache

OLLAMA_BLOB = (
    Path.home()
    / ".ollama/models/blobs/sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6"
)
MODEL = Path(os.environ["ARGUS_MODEL_PATH"]) if os.environ.get("ARGUS_MODEL_PATH") else OLLAMA_BLOB

has_lib = native.DEFAULT_LIB.exists()
needs_lib = pytest.mark.skipif(not has_lib, reason="needs the built libargus (native/README.md)")
needs_model = pytest.mark.skipif(
    not (has_lib and MODEL.exists()),
    reason="needs the built libargus and a GGUF embedding model (set ARGUS_MODEL_PATH)",
)

DIM = 4


class FakeStore:
    """In-memory stand-in for app.native.Index; `data` plays the role of the file on disk."""

    path = "fake-index"

    def __init__(self, data: dict):
        self.data = data

    def get_text(self, key):
        return self.data.get(("text", key))

    def get_vector(self, key):
        return self.data.get(("vec", key))

    def put_records(self, records):
        for key, text, vector in records:
            self.data[("text", key)] = text
            self.data[("vec", key)] = np.array(vector, dtype=np.float32)

    def meta_get(self, name):
        return self.data.get(("meta", name))

    def meta_set(self, name, value):
        self.data[("meta", name)] = value


@pytest.fixture(params=["fake", pytest.param("native", marks=needs_lib)])
def open_store(request, tmp_path, monkeypatch):
    """Returns a function that opens the (persistent) store; calling it again simulates a restart."""
    monkeypatch.setattr(settings, "argus_lib_path", None)
    opened = []
    data: dict = {}

    def open_():
        if request.param == "fake":
            return FakeStore(data)
        # Close the previous handle first: a restart does not keep the old process's handle open.
        while opened:
            opened.pop().close()
        store = native.Index(tmp_path / "idx.sqlite", DIM)
        opened.append(store)
        return store

    yield open_
    for store in opened:
        store.close()


@pytest.fixture
def model_file(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"model-a")
    return f


def vec(*xs):
    return np.array(xs, dtype=np.float32)


def test_saved_vectors_read_back_exactly(open_store, model_file):
    cache = VectorCache(open_store(), model_file)
    v = vec(0.1, -0.333333343, 1e-30, 7.0)
    cache.save({"pcb layout": v})
    got = cache.lookup(["pcb layout", "never saved"])
    assert list(got) == ["pcb layout"]
    assert got["pcb layout"].dtype == np.float32
    assert np.array_equal(got["pcb layout"], v)


def test_a_hash_collision_never_returns_the_other_phrases_vector(
    open_store, model_file, monkeypatch, caplog
):
    monkeypatch.setattr(vector_cache, "text_key", lambda text: 12345)
    cache = VectorCache(open_store(), model_file)
    cache.save({"alpha": vec(1, 0, 0, 0)})
    with caplog.at_level(logging.WARNING, logger="app.services.vector_cache"):
        assert cache.lookup(["beta"]) == {}
        cache.save({"beta": vec(0, 1, 0, 0)})  # not stored: the slot belongs to "alpha"
    assert "hash collision" in caplog.text
    assert cache.lookup(["beta"]) == {}
    assert np.array_equal(cache.lookup(["alpha"])["alpha"], vec(1, 0, 0, 0))


def test_a_collision_inside_one_batch_keeps_the_first_phrase(open_store, model_file, monkeypatch):
    monkeypatch.setattr(vector_cache, "text_key", lambda text: 7)
    cache = VectorCache(open_store(), model_file)
    cache.save({"alpha": vec(1, 0, 0, 0), "beta": vec(0, 1, 0, 0)})
    assert list(cache.lookup(["alpha", "beta"])) == ["alpha"]


def test_text_keys_are_stable_signed_64_bit_ids():
    key = vector_cache.text_key("machine learning")
    assert key == vector_cache.text_key("machine learning")
    assert key != vector_cache.text_key("machine learnin")
    assert -(2**63) <= key < 2**63


def test_opening_an_index_stamped_with_another_model_raises(open_store, model_file, tmp_path):
    VectorCache(open_store(), model_file)
    other = tmp_path / "other.gguf"
    other.write_bytes(b"model-b!")
    with pytest.raises(ModelMismatchError, match="different model.*Delete the index"):
        VectorCache(open_store(), other)


def test_the_model_is_hashed_once_while_its_size_and_mtime_are_unchanged(
    open_store, model_file, monkeypatch
):
    calls = []
    real = vector_cache._sha256_file
    monkeypatch.setattr(vector_cache, "_sha256_file", lambda p: calls.append(p) or real(p))
    for _ in range(3):
        VectorCache(open_store(), model_file)
    assert len(calls) == 1
    # Same content with a new mtime (e.g. the file was copied): re-hashed once, accepted.
    st = model_file.stat()
    os.utime(model_file, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    VectorCache(open_store(), model_file)
    VectorCache(open_store(), model_file)
    assert len(calls) == 2


def test_changed_content_with_a_new_mtime_is_caught(open_store, model_file):
    VectorCache(open_store(), model_file)
    model_file.write_bytes(b"model-b")  # same size, different weights
    st = model_file.stat()
    os.utime(model_file, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    with pytest.raises(ModelMismatchError):
        VectorCache(open_store(), model_file)


@needs_lib
def test_an_index_of_another_dimension_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "argus_lib_path", None)
    native.Index(tmp_path / "d.sqlite", DIM).close()
    with pytest.raises(ModelMismatchError, match="different dimension"):
        native.Index(tmp_path / "d.sqlite", DIM + 1)


# ---- embed_texts with a counting stub in place of the model ------------------------------------


@pytest.fixture
def stub_model(open_store, model_file, monkeypatch):
    """embed_texts on the native backend with a counting stub model and the persistent store under test."""
    calls: list[list[str]] = []

    def fake_embed(batch):
        calls.append(list(batch))
        rngs = [np.random.default_rng(vector_cache.text_key(t) % 2**32) for t in batch]
        return (
            np.stack([r.normal(size=DIM) for r in rngs]).astype(np.float32) * 3
        )  # not unit length

    monkeypatch.setattr(settings, "embeddings_backend", "native")
    monkeypatch.setattr(embeddings, "_embed_batch", fake_embed)
    monkeypatch.setattr(
        embeddings, "_open_persistent", lambda: VectorCache(open_store(), model_file)
    )
    monkeypatch.setattr(embeddings, "_embedding_cache", {})
    monkeypatch.setattr(embeddings, "_persistent", None)

    def restart():
        embeddings._embedding_cache.clear()
        embeddings._persistent = None

    return calls, restart


def test_cache_hit_vectors_equal_freshly_embedded_vectors_exactly(stub_model):
    calls, restart = stub_model
    phrases = ["pytorch", "pcb layout", "distributed systems design"]
    fresh = embeddings.embed_texts(phrases)
    assert np.allclose(np.linalg.norm(fresh, axis=1), 1.0, atol=1e-6)
    restart()
    from_disk = embeddings.embed_texts(phrases)
    assert from_disk.dtype == fresh.dtype
    assert np.array_equal(fresh, from_disk)
    assert len(calls) == 1


def test_a_restart_does_not_re_embed_known_phrases(stub_model):
    calls, restart = stub_model
    embeddings.embed_texts(["a", "b", "c"])
    restart()
    embeddings.embed_texts(["c", "b", "a", "a"])
    assert len(calls) == 1  # only the first process embedded anything
    restart()
    embeddings.embed_texts(["b", "d"])  # only the unseen phrase reaches the model
    assert calls[-1] == ["d"]


def test_the_in_process_dict_answers_before_the_index(stub_model, monkeypatch):
    calls, _ = stub_model
    embeddings.embed_texts(["a"])
    monkeypatch.setattr(embeddings, "_open_persistent", lambda: pytest.fail("index consulted"))
    monkeypatch.setattr(embeddings, "_persistent", None)
    embeddings.embed_texts(["a"])
    assert len(calls) == 1


def test_ollama_backend_never_touches_the_persistent_index(monkeypatch):
    monkeypatch.setattr(settings, "embeddings_backend", "ollama")
    monkeypatch.setattr(embeddings, "_embedding_cache", {})
    monkeypatch.setattr(embeddings, "_open_persistent", lambda: pytest.fail("index opened"))
    monkeypatch.setattr(embeddings, "_embed_batch", lambda b: [[3.0, 4.0] for _ in b])
    assert np.allclose(embeddings.embed_texts(["x"]), [[0.6, 0.8]])


# ---- configuration and missing pieces ----------------------------------------------------------


def test_default_backend_is_native():
    assert settings.embeddings_backend == "native"


@needs_lib
def test_native_backend_without_a_model_fails_loudly_with_the_fetch_command(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "embeddings_backend", "native")
    monkeypatch.setattr(settings, "argus_lib_path", None)
    monkeypatch.setattr(settings, "argus_model_path", str(tmp_path / "missing.gguf"))
    monkeypatch.setattr(embeddings, "_embedding_cache", {})
    monkeypatch.setattr(embeddings, "_persistent", None)
    with pytest.raises(NativeError, match=r"python3 native/tools/fetch_model\.py"):
        embeddings.embed_texts(["anything"])


def test_native_backend_with_a_missing_library_fails_loudly_with_the_build_command(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "embeddings_backend", "native")
    monkeypatch.setattr(settings, "argus_lib_path", str(tmp_path / "libargus.dylib"))
    monkeypatch.setattr(native, "_lib", None)
    monkeypatch.setattr(embeddings, "_embedding_cache", {})
    monkeypatch.setattr(embeddings, "_persistent", None)
    with pytest.raises(NativeError, match=r"libargus not found.*cmake -S native"):
        embeddings.embed_texts(["anything"])


# ---- the real library and model ----------------------------------------------------------------


@pytest.fixture
def real_native(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "embeddings_backend", "native")
    monkeypatch.setattr(settings, "argus_model_path", str(MODEL))
    monkeypatch.setattr(settings, "argus_lib_path", None)
    monkeypatch.setattr(settings, "argus_index_path", str(tmp_path / "real.sqlite"))
    monkeypatch.setattr(embeddings, "_embedding_cache", {})
    monkeypatch.setattr(embeddings, "_persistent", None)
    yield
    embeddings.close()  # Metal asserts if a model is loaded at process exit; also keeps tests isolated


@needs_model
def test_native_vectors_are_unit_length_with_the_models_dimension(real_native):
    v = embeddings.embed_texts(["PyTorch", "distributed systems design"])
    assert v.shape == (2, 768)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


@needs_model
def test_native_embeddings_capture_meaning(real_native):
    v = embeddings.embed_texts(["machine learning", "deep learning", "pastry baking"])
    assert v[0] @ v[1] > v[0] @ v[2]


@needs_model
def test_real_model_restart_reads_identical_vectors_without_embedding(real_native, monkeypatch):
    phrases = ["machine learning", "deep learning", "pastry baking"]
    fresh = embeddings.embed_texts(phrases)
    embeddings.close()
    embeddings._embedding_cache.clear()
    monkeypatch.setattr(
        embeddings, "_embed_batch", lambda b: pytest.fail(f"embedded after restart: {b}")
    )
    again = embeddings.embed_texts(phrases)
    assert np.array_equal(fresh, again)
