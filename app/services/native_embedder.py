"""In-process embeddings through libargus (llama.cpp), the native alternative to Ollama.

Same model weights as the Ollama path (nomic-embed-text), so vectors agree to ~6 decimals
(native/reports/ollama_agreement.md), without a separate server to install.
Selected with EMBEDDINGS_BACKEND=native; needs ARGUS_MODEL_PATH (a GGUF embedding model) and the
built library (ARGUS_LIB_PATH, default: the repo's native/build-release).

Raises loudly if the library or model is missing -- no silent fallback to Ollama, since which
runtime produced a vector is something the caller should know.
"""

import atexit
import ctypes
import threading
from pathlib import Path

import numpy as np

from app.config import settings

DEFAULT_LIB = Path(__file__).resolve().parents[2] / "native/build-release/libargus.dylib"
_BATCH = 250

_lock = threading.Lock()
_state: tuple[ctypes.CDLL, ctypes.c_void_p, int] | None = None


def _status(code: int, what: str) -> None:
    if code != 0:
        raise RuntimeError(f"libargus: {what} failed with status {code}")


def _load() -> tuple[ctypes.CDLL, ctypes.c_void_p, int]:
    global _state
    if _state is not None:
        return _state
    lib_path = Path(settings.argus_lib_path) if settings.argus_lib_path else DEFAULT_LIB
    if not lib_path.exists():
        raise RuntimeError(f"libargus not found at {lib_path}; build native or set ARGUS_LIB_PATH")
    if not settings.argus_model_path or not Path(settings.argus_model_path).exists():
        raise RuntimeError(
            "EMBEDDINGS_BACKEND=native needs ARGUS_MODEL_PATH pointing at a GGUF embedding model"
        )

    lib = ctypes.CDLL(str(lib_path))
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

    handle = ctypes.c_void_p()
    _status(
        lib.argus_embedder_open(settings.argus_model_path.encode(), 0, ctypes.byref(handle)),
        "loading the model",
    )
    dim = ctypes.c_size_t()
    _status(lib.argus_embedder_dim(handle, ctypes.byref(dim)), "reading the embedding dimension")
    _state = (lib, handle, dim.value)
    # llama.cpp's Metal backend asserts if the process exits with a model still loaded.
    atexit.register(close)
    return _state


def close() -> None:
    global _state
    with _lock:
        if _state is not None:
            lib, handle, _ = _state
            lib.argus_embedder_close(handle)
            _state = None


def embed(texts: list[str]) -> np.ndarray:
    """(len(texts), dim) float32, L2-normalized, in input order."""
    with _lock:
        lib, handle, dim = _load()
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for start in range(0, len(texts), _BATCH):
            part = [t.encode() for t in texts[start : start + _BATCH]]
            arr = (ctypes.c_char_p * len(part))(*part)
            lens = (ctypes.c_size_t * len(part))(*[len(p) for p in part])
            buf = (ctypes.c_float * (len(part) * dim))()
            _status(
                lib.argus_embed_batch(handle, arr, lens, len(part), buf, dim, None),
                "embedding a batch",
            )
            out[start : start + len(part)] = np.frombuffer(buf, dtype=np.float32).reshape(
                len(part), dim
            )
        return out
