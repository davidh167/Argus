"""End-to-end phrase-embedding wall time, the way market_demand calls it (embeddings.embed_texts over the
distinct stage2_key_skills phrases): Ollama vs native, and a restart with vs without the persistent cache.

Scenarios (each timed `--repeats` times, reported as median and range):
  ollama cold                      EMBEDDINGS_BACKEND=ollama, nothing cached in the process
  native cold, no persistence      model loaded from scratch, nothing read from or written to the index
  native cold, persisting          fresh index file; every vector is embedded and written
  native restart, persistence      the process state is dropped (model and index closed and reopened,
                                   in-process dict cleared) and the same phrases are requested again
  native restart, no persistence   the same restart without the index: everything is embedded again
  in-process warm                  the same phrases again with the in-process dict filled
"Restart" is simulated inside one interpreter: it includes reloading the model, reopening the index
and verifying the model identity, but not Python start-up or imports.

Input: one phrase per line. The owner exports the real list (read-only on the DB) with
  PYTHONPATH=. poetry run python research/native-engine/embeddings/export_phrases.py
and runs, for example,
  PYTHONPATH=. poetry run python research/native-engine/embeddings/bench_embeddings.py --repeats 3
Ollama must be running for the ollama scenario (`--skip-ollama` leaves it out). `--synthetic N` generates N
phrases instead, and `--stub-embed` replaces the model with a hash-based stub (no Ollama, no model; needs only
the built library), which only exercises the harness: its timings mean nothing.
"""

import argparse
import hashlib
import itertools
import platform
import statistics
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from app import native
from app.config import settings
from app.services import embeddings
from app.services.vector_cache import VectorCache

DEFAULT_PHRASES = Path(__file__).resolve().parents[1] / "data" / "embedding_phrases.txt"
STUB_DIM = 768
_WORDS = (
    "distributed systems design streaming data pipelines model serving feature stores vector search "
    "compiler optimization embedded firmware signal processing circuit layout robotics perception "
    "kubernetes terraform observability incident response technical writing product analytics "
    "experiment design causal inference time series forecasting reinforcement learning"
).split()
_WORDS = tuple(dict.fromkeys(_WORDS))


def synthetic_phrases(n: int) -> list[str]:
    """n distinct 2-4 word phrases, deterministic."""
    phrases: list[str] = []
    for length in (2, 3, 4):
        for combo in itertools.product(_WORDS, repeat=length):
            phrases.append(" ".join(combo))
            if len(phrases) == n:
                return phrases
    raise ValueError(f"cannot generate {n} synthetic phrases")


def read_phrases(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Export the phrases first (reads the real DB, run it yourself): "
            "PYTHONPATH=. poetry run python research/native-engine/embeddings/export_phrases.py, "
            "or pass --phrases FILE or --synthetic N."
        )
    seen: dict[str, None] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            seen.setdefault(line.strip())
    return list(seen)


def stub_embed(batch: list[str]) -> np.ndarray:
    """Deterministic pseudo-embeddings: stands in for the model when only the harness is under test."""
    out = np.empty((len(batch), STUB_DIM), dtype=np.float32)
    for i, text in enumerate(batch):
        seed = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")
        out[i] = np.random.default_rng(seed).normal(size=STUB_DIM)
    return out


class NoPersistence:
    """Drop-in for VectorCache that remembers nothing, to time the native backend without the index."""

    def lookup(self, texts: list[str]) -> dict[str, np.ndarray]:
        return {}

    def save(self, items: dict[str, np.ndarray]) -> None:
        pass


class Harness:
    def __init__(self, workdir: Path, stub: bool) -> None:
        self.workdir = workdir
        self.stub = stub
        self.index_file = workdir / "bench_index.sqlite"
        self._stub_indexes: list[native.Index] = []
        self._stub_model = workdir / "stub-model.bin"
        self._stub_model.write_bytes(b"stub model identity")
        settings.argus_index_path = str(self.index_file)
        if stub:
            embeddings._embed_batch = stub_embed  # type: ignore[assignment]
            embeddings._open_persistent = self._open_stub_persistent  # type: ignore[assignment]

    def _open_stub_persistent(self) -> VectorCache:
        index = native.Index(self.index_file, STUB_DIM)
        self._stub_indexes.append(index)
        return VectorCache(index, self._stub_model)

    def restart(self) -> None:
        """Forget everything a new process would not have: the dict, the open index, the loaded model."""
        for index in self._stub_indexes:
            index.close()
        self._stub_indexes.clear()
        embeddings.close()
        embeddings._embedding_cache.clear()

    def fresh_index(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.index_file) + suffix).unlink(missing_ok=True)


def timed(fn: Callable[[], object]) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def run(phrases: list[str], repeats: int, include_ollama: bool, harness: Harness) -> list[dict]:
    results: list[dict] = []
    real_persistent_cache = embeddings._persistent_cache

    def embed_all() -> object:
        return embeddings.embed_texts(phrases)

    def scenario(name: str, backend: str, setup: Callable[[], None], persistence: bool) -> None:
        settings.embeddings_backend = backend  # type: ignore[assignment]
        times = []
        for _ in range(repeats):
            harness.restart()
            setup()
            embeddings._persistent_cache = (  # type: ignore[assignment]
                real_persistent_cache if persistence else (lambda: NoPersistence())  # type: ignore[return-value]
            )
            times.append(timed(embed_all))
        results.append({"scenario": name, "times": times})

    if include_ollama:
        scenario("ollama cold", "ollama", lambda: None, persistence=False)
    scenario("native cold, no persistence", "native", lambda: None, persistence=False)
    scenario("native cold, persisting", "native", harness.fresh_index, persistence=True)

    # The index is filled by the scenario above; restarts read it.
    scenario("native restart, persistence", "native", lambda: None, persistence=True)
    scenario("native restart, no persistence", "native", lambda: None, persistence=False)

    settings.embeddings_backend = "native"  # type: ignore[assignment]
    embeddings._persistent_cache = real_persistent_cache  # type: ignore[assignment]
    harness.restart()
    embed_all()
    results.append(
        {"scenario": "in-process warm", "times": [timed(embed_all) for _ in range(repeats)]}
    )
    harness.restart()
    return results


def report(results: list[dict], n_phrases: int, repeats: int, stub: bool) -> str:
    lines = [
        "# Embedding benchmark"
        + (" (STUB EMBEDDER: harness check only, timings are meaningless)" if stub else ""),
        "",
        f"- phrases: {n_phrases} distinct; repeats per scenario: {repeats}",
        f"- machine: {platform.platform()}, Python {platform.python_version()}",
        "",
        "| scenario | median s | min s | max s |",
        "|---|---|---|---|",
    ]
    for r in results:
        t = r["times"]
        lines.append(
            f"| {r['scenario']} | {statistics.median(t):.2f} | {min(t):.2f} | {max(t):.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--phrases", type=Path, default=DEFAULT_PHRASES, help="one phrase per line")
    src.add_argument("--synthetic", type=int, metavar="N", help="generate N phrases instead")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip-ollama", action="store_true")
    ap.add_argument(
        "--stub-embed", action="store_true", help="hash-based stub instead of any model"
    )
    ap.add_argument("--out", type=Path, help="also write the markdown report here")
    args = ap.parse_args()

    phrases = synthetic_phrases(args.synthetic) if args.synthetic else read_phrases(args.phrases)
    skip_ollama = args.skip_ollama or args.stub_embed
    with tempfile.TemporaryDirectory() as tmp:
        harness = Harness(Path(tmp), args.stub_embed)
        text = report(
            run(phrases, args.repeats, not skip_ollama, harness),
            len(phrases),
            args.repeats,
            args.stub_embed,
        )
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
