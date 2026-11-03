"""Python access to libargus (native/), the in-process embedding, vector-index and similarity core.

Owns everything that touches the C library through ctypes: finding and loading it, turning
non-zero statuses into exceptions that carry the library's own error text, and the process-wide
handles (one embedder, one persistent vector index) plus the in-memory similarity index. Nothing
else in the app calls ctypes.

Nothing here falls back to anything: a missing library, model or index mismatch raises with the
command that fixes it, since which runtime and model produced a vector is something the caller
must know.
"""

import atexit
import ctypes
import sys
import threading
import weakref
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from app.config import settings

REPO_ROOT = Path(__file__).resolve().parents[2]
# The release build directory from native/README.md; the shared library's suffix is the platform's.
DEFAULT_LIB = REPO_ROOT / (
    "native/build-release/libargus" + (".dylib" if sys.platform == "darwin" else ".so")
)
# Next to the model that native/tools/fetch_model.py downloads (its DEFAULT_DEST and file name).
APP_DATA_DIR = Path.home() / ".local/share/job-search-tool"
DEFAULT_MODEL = APP_DATA_DIR / "models/nomic-embed-text-v1.5.f16.gguf"
DEFAULT_INDEX = APP_DATA_DIR / "embeddings.sqlite"

_BATCH = 250
_TEXT_BUFFER = 256  # initial size for reading a stored text back; retried at the real size if short
_STATUS_NAMES = {1: "invalid argument", 2: "storage error", 3: "out of memory", 4: "model error"}

_BUILD_HINT = (
    "build it with `cmake -S native -B native/build-release -G Ninja && "
    "cmake --build native/build-release` (see native/README.md), or set ARGUS_LIB_PATH"
)
_MODEL_HINT = (
    "download it (274 MB) with `python3 native/tools/fetch_model.py`, or set ARGUS_MODEL_PATH"
)
_NO_LLAMA_HINT = (
    "this libargus was built without llama.cpp (-DARGUS_WITH_LLAMA=OFF), so it has no embedder; "
    "rebuild with the default -DARGUS_WITH_LLAMA=ON (native/README.md)"
)
_OLLAMA_HINT = "or set EMBEDDINGS_BACKEND=ollama to use a local Ollama server instead"


class NativeError(RuntimeError):
    """A libargus call failed, or the library or model it needs is missing."""


class ModelMismatchError(NativeError):
    """The persistent index was built with a different embedding model than the current one."""


def lib_path() -> Path:
    return Path(settings.argus_lib_path) if settings.argus_lib_path else DEFAULT_LIB


def model_path() -> Path:
    return Path(settings.argus_model_path) if settings.argus_model_path else DEFAULT_MODEL


def index_path() -> Path:
    return Path(settings.argus_index_path) if settings.argus_index_path else DEFAULT_INDEX


class _Hit(ctypes.Structure):
    """argus_hit."""

    _fields_ = [("id", ctypes.c_int64), ("score", ctypes.c_double)]


_lock = threading.RLock()
_lib: ctypes.CDLL | None = None
_embedder: "Embedder | None" = None
_index: "Index | None" = None


def _check(status: int, what: str, last_error: Callable[[], str] | None = None) -> None:
    if status != 0:
        detail = f": {last_error()}" if last_error else ""
        name = _STATUS_NAMES.get(status, f"status {status}")
        raise NativeError(f"libargus: {what} failed ({name}){detail}")


def load_lib() -> ctypes.CDLL:
    global _lib
    with _lock:
        if _lib is not None:
            return _lib
        path = lib_path()
        if not path.exists():
            raise NativeError(f"libargus not found at {path}; {_BUILD_HINT}; {_OLLAMA_HINT}.")
        lib = ctypes.CDLL(str(path))
        vp, sz, i64 = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64
        pp, pi = ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int)
        pf, ps = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_size_t)
        signatures = {
            "argus_open_with_vectors": [ctypes.c_char_p, sz, pp],
            "argus_close": [vp],
            "argus_upsert_records": [
                vp,
                sz,
                ctypes.POINTER(i64),
                ctypes.POINTER(ctypes.c_char_p),
                ps,
                pf,
                sz,
            ],
            "argus_get_vector": [vp, i64, pf, sz, pi],
            "argus_get_text": [vp, i64, ctypes.c_char_p, sz, ps, pi],
            "argus_meta_set": [vp, ctypes.c_char_p, ctypes.c_char_p],
            "argus_meta_get": [vp, ctypes.c_char_p, ctypes.c_char_p, sz, ps, pi],
            "argus_last_error": [vp],
            "argus_sim_create": [pp],
            "argus_sim_free": [vp],
            "argus_sim_add": [vp, i64, ctypes.c_char_p, sz],
            "argus_sim_finalize": [vp],
            "argus_sim_count": [vp, ps],
            "argus_sim_topk": [vp, i64, sz, ctypes.POINTER(_Hit), ps, pi],
            "argus_sim_last_error": [vp],
            "argus_text_tokenize": [ctypes.c_char_p, sz, ctypes.c_char_p, sz, ps],
            "argus_unicode_probe": [
                ctypes.c_uint32,
                pi,
                ctypes.POINTER(ctypes.c_uint32),
                sz,
                ps,
            ],
        }
        for name, argtypes in signatures.items():
            getattr(lib, name).argtypes = argtypes
        lib.argus_sim_last_error.restype = ctypes.c_char_p
        lib.argus_unicode_version.restype = ctypes.c_char_p
        lib.argus_icu_version.restype = ctypes.c_char_p
        lib.argus_last_error.restype = ctypes.c_char_p
        _lib = lib
        return lib


def _bind_embedder(lib: ctypes.CDLL) -> None:
    """Sets the embedder entry points' signatures; a library built without llama.cpp lacks them."""
    if not hasattr(lib, "argus_embedder_open"):
        raise NativeError(f"{lib_path()}: {_NO_LLAMA_HINT}.")
    vp, sz = ctypes.c_void_p, ctypes.c_size_t
    signatures = {
        "argus_embedder_open": [ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)],
        "argus_embedder_close": [vp],
        "argus_embedder_dim": [vp, ctypes.POINTER(sz)],
        "argus_embed_batch": [
            vp,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(sz),
            sz,
            ctypes.POINTER(ctypes.c_float),
            sz,
            ctypes.POINTER(ctypes.c_uint8),
        ],
        "argus_embedder_last_error": [vp],
    }
    for name, argtypes in signatures.items():
        getattr(lib, name).argtypes = argtypes
    lib.argus_embedder_last_error.restype = ctypes.c_char_p


class Embedder:
    """A loaded GGUF embedding model."""

    def __init__(self, path: Path) -> None:
        self._lib = load_lib()
        _bind_embedder(self._lib)
        self._handle = ctypes.c_void_p()
        _check(
            self._lib.argus_embedder_open(str(path).encode(), 0, ctypes.byref(self._handle)),
            f"loading the model {path}",
        )
        dim = ctypes.c_size_t()
        _check(
            self._lib.argus_embedder_dim(self._handle, ctypes.byref(dim)),
            "reading the embedding dimension",
        )
        self.dim = dim.value

    def _error(self) -> str:
        return (self._lib.argus_embedder_last_error(self._handle) or b"").decode()

    def embed(self, texts: list[str]) -> np.ndarray:
        """(len(texts), dim) float32, L2-normalized by the library, in input order."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(texts), _BATCH):
            part = [t.encode() for t in texts[start : start + _BATCH]]
            arr = (ctypes.c_char_p * len(part))(*part)
            lens = (ctypes.c_size_t * len(part))(*[len(p) for p in part])
            buf = (ctypes.c_float * (len(part) * self.dim))()
            _check(
                self._lib.argus_embed_batch(
                    self._handle, arr, lens, len(part), buf, self.dim, None
                ),
                "embedding a batch",
                self._error,
            )
            out[start : start + len(part)] = np.frombuffer(buf, dtype=np.float32).reshape(
                len(part), self.dim
            )
        return out

    def close(self) -> None:
        self._lib.argus_embedder_close(self._handle)


class Index:
    """A persistent index file used as a key -> (text, vector) store, plus string metadata."""

    def __init__(self, path: Path, dim: int) -> None:
        self.path = str(path)
        self.dim = dim
        self._lib = load_lib()
        self._handle = ctypes.c_void_p()
        path.parent.mkdir(parents=True, exist_ok=True)
        status = self._lib.argus_open_with_vectors(
            self.path.encode(), dim, ctypes.byref(self._handle)
        )
        if status == 1:  # the only way a non-null path and a positive dim are rejected
            raise ModelMismatchError(
                f"{self.path} holds vectors of a different dimension than the current model's "
                f"({dim}); it was built with another embedding model. Delete it to rebuild: "
                f"rm {self.path}*"
            )
        _check(status, f"opening the index {self.path}")

    def _error(self) -> str:
        return (self._lib.argus_last_error(self._handle) or b"").decode()

    def put_records(self, records: list[tuple[int, str, np.ndarray]]) -> None:
        """Stores (key, text, vector) records in one transaction; all are written or none."""
        n = len(records)
        encoded = [text.encode() for _, text, _ in records]
        keys = (ctypes.c_int64 * n)(*[key for key, _, _ in records])
        texts = (ctypes.c_char_p * n)(*encoded)
        lens = (ctypes.c_size_t * n)(*[len(e) for e in encoded])
        vecs = np.ascontiguousarray([v for _, _, v in records], dtype=np.float32).reshape(
            n, self.dim
        )
        _check(
            self._lib.argus_upsert_records(
                self._handle,
                n,
                keys,
                texts,
                lens,
                vecs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.dim,
            ),
            "storing records",
            self._error,
        )

    def get_vector(self, key: int) -> np.ndarray | None:
        out = np.zeros(self.dim, dtype=np.float32)
        found = ctypes.c_int()
        _check(
            self._lib.argus_get_vector(
                self._handle,
                key,
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.dim,
                ctypes.byref(found),
            ),
            "reading a vector",
            self._error,
        )
        return out if found.value else None

    def _read_string(self, call: Callable[..., int], what: str, *args: object) -> str | None:
        cap = _TEXT_BUFFER
        while True:
            buf = ctypes.create_string_buffer(cap)
            length, found = ctypes.c_size_t(), ctypes.c_int()
            _check(
                call(self._handle, *args, buf, cap, ctypes.byref(length), ctypes.byref(found)),
                what,
                self._error,
            )
            if not found.value:
                return None
            if length.value <= cap:
                return buf.raw[: length.value].decode()
            cap = length.value

    def get_text(self, key: int) -> str | None:
        return self._read_string(self._lib.argus_get_text, "reading a text", key)

    def meta_get(self, name: str) -> str | None:
        return self._read_string(self._lib.argus_meta_get, "reading metadata", name.encode())

    def meta_set(self, name: str, value: str) -> None:
        _check(
            self._lib.argus_meta_set(self._handle, name.encode(), value.encode()),
            "storing metadata",
            self._error,
        )

    def close(self) -> None:
        self._lib.argus_close(self._handle)


class Similarity:
    """An in-memory TF-IDF similarity index over a fixed corpus (argus_sim_*), the native twin of
    app.services.relevance's vectors + cosine scan. Documents are tokenized, weighted and normalized
    at construction; top_k then ranks every other document against one of them. Tie order is the
    order of `documents`, as Python's stable sort over the same dict would give."""

    def __init__(self, documents: Iterable[tuple[int, str]]) -> None:
        self._lib = load_lib()
        self._handle = ctypes.c_void_p()
        _check(
            self._lib.argus_sim_create(ctypes.byref(self._handle)), "creating a similarity index"
        )
        # Freed when this object is collected, e.g. when relevance's cache replaces it.
        weakref.finalize(self, self._lib.argus_sim_free, self._handle)
        for key, text in documents:
            raw = text.encode()
            _check(
                self._lib.argus_sim_add(self._handle, key, raw, len(raw)),
                f"adding document {key}",
                self._error,
            )
        _check(self._lib.argus_sim_finalize(self._handle), "building the index", self._error)
        self._size = len(self)

    def _error(self) -> str:
        return (self._lib.argus_sim_last_error(self._handle) or b"").decode()

    def __len__(self) -> int:
        count = ctypes.c_size_t()
        _check(self._lib.argus_sim_count(self._handle, ctypes.byref(count)), "counting documents")
        return count.value

    def top_k(self, key: int, k: int) -> list[tuple[int, float]] | None:
        """The k most similar other documents as (key, cosine), best first; None if key is unknown."""
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        k = min(k, self._size)  # never size the result buffer by a caller's huge limit
        hits = (_Hit * max(k, 1))()
        n, found = ctypes.c_size_t(), ctypes.c_int()
        _check(
            self._lib.argus_sim_topk(
                self._handle, key, k, hits, ctypes.byref(n), ctypes.byref(found)
            ),
            "ranking similar documents",
            self._error,
        )
        return [(hits[i].id, hits[i].score) for i in range(n.value)] if found.value else None


def tokenize(text: str) -> list[str]:
    """The library's tokenizer: Python's re.findall(r"\\w+", text.lower()). A parity hook for tests."""
    raw = text.encode()
    lib = load_lib()
    cap = len(raw) * 4 + 16  # lowercasing can grow text; the call reports the true size either way
    while True:
        buf = ctypes.create_string_buffer(cap)
        needed = ctypes.c_size_t()
        _check(lib.argus_text_tokenize(raw, len(raw), buf, cap, ctypes.byref(needed)), "tokenizing")
        if needed.value <= cap:
            return buf.raw[: needed.value].decode().split("\n")[:-1]
        cap = needed.value


def unicode_probe(code_point: int) -> tuple[bool, str]:
    """(is a \\w character, full lowercase mapping) of one code point, per the library. A parity hook."""
    lib = load_lib()
    is_word = ctypes.c_int()
    out = (ctypes.c_uint32 * 8)()
    n = ctypes.c_size_t()
    _check(
        lib.argus_unicode_probe(code_point, ctypes.byref(is_word), out, 8, ctypes.byref(n)),
        f"probing U+{code_point:04X}",
    )
    return bool(is_word.value), "".join(chr(out[i]) for i in range(n.value))


def unicode_versions() -> tuple[str, str]:
    """(Unicode version, ICU version) the loaded library runs against."""
    lib = load_lib()
    return lib.argus_unicode_version().decode(), lib.argus_icu_version().decode()


def embedder() -> Embedder:
    """The process-wide embedder, loading the model on first use."""
    global _embedder
    if _embedder is None:
        load_lib()
        path = model_path()
        if not path.exists():
            raise NativeError(
                f"Embedding model not found at {path}; {_MODEL_HINT}; {_OLLAMA_HINT}."
            )
        with _lock:
            if _embedder is None:
                _embedder = Embedder(path)
                # llama.cpp's Metal backend asserts if the process exits with a model still loaded.
                atexit.register(close)
    return _embedder


def vector_index() -> Index:
    """The process-wide persistent vector index, sized to the current model's dimension."""
    global _index
    if _index is None:
        dim = embedder().dim
        with _lock:
            if _index is None:
                _index = Index(index_path(), dim)
    return _index


def close() -> None:
    """Releases the model and the index; the next use reopens them."""
    global _embedder, _index
    with _lock:
        if _index is not None:
            _index.close()
            _index = None
        if _embedder is not None:
            _embedder.close()
            _embedder = None
