"""Phase 2 cases: the native similarity engine against the Python implementation, same corpus, same machine.

Two levels, both timed with bench_cases.measure (warmup runs, then timed runs, wall clock):

- through the app: relevance.find_similar_postings under each engine (settings.similarity_engine), cold
  (cache cleared: DB fetch + index build + query) and warm (cache hit: cache-key query + query + ORM fetch);
- the engine alone, no database: building the index from (id, text) pairs, one top-10 query, and the cost of
  a ctypes call with no work behind it.

Memory is measured in fresh subprocesses (bench_memory.py), one per engine and repeat.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy import select

import bench_parity
from app import native
from app.models.posting import Posting
from app.services import relevance
from bench_cases import SessionFactory, measure, similarity_engine, target_posting_ids
from bench_report import Timing, fmt_ms, percentile, timings_table

HERE = Path(__file__).resolve().parent
MEMORY_REPEATS = 3


def fetch_documents(session_factory: SessionFactory) -> list[tuple[int, str]]:
    """(id, similarity text) in the row order _get_similarity_index feeds the engines."""
    with session_factory() as db:
        rows = list(db.execute(select(Posting.id, Posting.title, Posting.normalized_jd_text)))
    return [(pid, relevance._similarity_text(title, normalized)) for pid, title, normalized in rows]


def run_timings(
    session_factory: SessionFactory,
    documents: list[tuple[int, str]],
    *,
    runs: int,
    warmup: int,
    query_runs: int,
) -> list[Timing]:
    targets = target_posting_ids(session_factory)
    counter = {"i": 0}

    def next_target() -> int:
        counter["i"] += 1
        return targets[counter["i"] % len(targets)]

    def find_similar() -> list:
        with session_factory() as db:
            result = relevance.find_similar_postings(db, next_target(), limit=10)
        assert result is not None
        return result

    timings: list[Timing] = []
    for engine in ("python", "native"):
        with similarity_engine(engine):
            t, _ = measure(
                f"find_similar_postings ({engine}, cold)",
                "similarity cache cleared before each call: DB fetch + index build + query + top-10 ORM fetch",
                find_similar,
                runs=runs,
                warmup=warmup,
                before=relevance._similarity_index_cache.clear,
            )
            timings.append(t)
            relevance._similarity_index_cache.clear()
            find_similar()
            t, _ = measure(
                f"find_similar_postings ({engine}, warm)",
                "cache hit: cache-key query + query + top-10 ORM fetch",
                find_similar,
                runs=runs,
                warmup=warmup,
            )
            timings.append(t)
    relevance._similarity_index_cache.clear()

    t_py_build, vectors = measure(
        "index build from texts (python)",
        "relevance.python_similarity_vectors: tokenize + df/idf + tf-idf dict per document",
        lambda: relevance.python_similarity_vectors(documents),
        runs=runs,
        warmup=warmup,
    )
    t_nat_build, index = measure(
        "index build from texts (native)",
        "native.Similarity: ICU tokenize + vocabulary + idf + CSR weights, from the same (id, text) pairs",
        lambda: native.Similarity(documents),
        runs=runs,
        warmup=warmup,
    )
    timings += [t_py_build, t_nat_build]

    ids = [pid for pid, _ in documents]
    cycle = {"i": 0}

    def next_doc() -> int:
        cycle["i"] += 1
        return targets[cycle["i"] % len(targets)]

    t_py_query, _ = measure(
        "top-10 query (python)",
        "relevance.python_top_k over the prebuilt vectors: cosine of one target against every document, sorted",
        lambda: relevance.python_top_k(vectors, next_doc(), 10),
        runs=runs,
        warmup=warmup,
    )
    t_nat_query, _ = measure(
        "top-10 query (native)",
        "Similarity.top_k(id, 10): one ctypes call, scan of every document, result list built in Python",
        lambda: index.top_k(next_doc(), 10),
        runs=query_runs,
        warmup=warmup,
    )
    t_nat_noop, _ = measure(
        "ctypes call, no work (native)",
        "Similarity.top_k(id, 0): lookup of the id and return; the wrapper's argument and result handling "
        "plus the ctypes crossing, no scan",
        lambda: index.top_k(next_doc(), 0),
        runs=query_runs,
        warmup=warmup,
    )
    timings += [t_py_query, t_nat_query, t_nat_noop]
    assert len(ids) == len(index)
    return timings


def memory_runs(corpus_args: list[str]) -> dict[str, list[dict]]:
    """Index memory per engine, one fresh process per sample (bench_memory.py); corpus_args are its
    corpus selection flags (--n N or --snapshot DIR)."""
    out: dict[str, list[dict]] = {}
    for engine in ("python", "native"):
        out[engine] = []
        for _ in range(MEMORY_REPEATS):
            proc = subprocess.run(
                [sys.executable, str(HERE / "bench_memory.py"), "--engine", engine, *corpus_args],
                capture_output=True,
                text=True,
                check=True,
                env={**os.environ, "PYTHONPATH": str(HERE.parents[1])},
            )
            out[engine].append(json.loads(proc.stdout))
            print(f"  memory {engine}: {out[engine][-1]}", file=sys.stderr, flush=True)
    return out


def parity_gate(documents: list[tuple[int, str]], sample: int) -> bench_parity.ParityResult:
    return bench_parity.compare(documents, k=10, sample=sample)


def _median(values: list[float]) -> float:
    return percentile(values, 50)


def _ratio(py: float, nat: float) -> str:
    return f"{py / nat:,.1f}x" if nat > 0 else "n/a"


def comparison_table(timings: list[Timing]) -> str:
    by = {t.name: t for t in timings}
    pairs = [
        ("find_similar, cold (DB fetch + build + query)", "find_similar_postings ({}, cold)"),
        ("find_similar, warm (cache hit)", "find_similar_postings ({}, warm)"),
        ("index build from texts", "index build from texts ({})"),
        ("top-10 query, engine only", "top-10 query ({})"),
    ]
    lines = [
        "| What | Python median ms | Native median ms | Median ratio | Python p95 ms | Native p95 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, pattern in pairs:
        py, nat = by[pattern.format("python")].samples_ms, by[pattern.format("native")].samples_ms
        lines.append(
            f"| {label} | {fmt_ms(_median(py))} | {fmt_ms(_median(nat))} "
            f"| {_ratio(_median(py), _median(nat))} | {fmt_ms(percentile(py, 95))} "
            f"| {fmt_ms(percentile(nat, 95))} |"
        )
    return "\n".join(lines)


def memory_table(samples: dict[str, list[dict]]) -> str:
    lines = [
        "| Engine | Index RSS delta MB (median of samples) | Samples MB |",
        "|---|---:|---|",
    ]
    for engine, runs in samples.items():
        deltas = [r["delta_mb"] for r in runs]
        lines.append(
            f"| {engine} | {_median(deltas):.1f} | {', '.join(f'{d:.1f}' for d in deltas)} |"
        )
    py = _median([r["delta_mb"] for r in samples["python"]])
    nat = _median([r["delta_mb"] for r in samples["native"]])
    lines += ["", f"Python / native memory ratio: {_ratio(py, nat)}."]
    return "\n".join(lines)


def corpus_section(
    name: str,
    n: int,
    timings: list[Timing],
    memory: dict[str, list[dict]],
    parity: bench_parity.ParityResult,
    notes: dict[str, str],
) -> str:
    runs = min(len(t.samples_ms) for t in timings)
    parity_line = (
        f"{parity.n_targets} query postings (seeded sample) x top-10 against the Python implementation fed in "
        f"the same order: {parity.n_id_mismatch} id/order mismatches, {parity.n_score_mismatch} score mismatches "
        f"(tolerance {bench_parity.SCORE_TOLERANCE:g}), max |score diff| {parity.max_abs_score_diff:.3g}, "
        f"{parity.n_exact_score_matches} of {parity.n_hits} scores bit-identical"
    )
    parts = [
        f"## Corpus: {name} (N = {n:,})",
        "",
        "\n".join(f"- {k}: {v}" for k, v in notes.items()),
        f"- parity gate (run before timing): {parity_line}",
        f"- fewest timed runs in any case: {runs}",
        "",
        "### Native vs Python",
        "",
        comparison_table(timings),
        "",
        "### All timings",
        "",
        timings_table(timings),
        "",
        "Case definitions:",
        "",
        "\n".join(f"- {t.name}: {t.what}" for t in timings),
        "",
        "### Memory",
        "",
        memory_table(memory),
    ]
    return "\n".join(parts)


def render_report(env: dict[str, str], sections: list[str]) -> str:
    unicode_version, icu_version = native.unicode_versions()
    env = {
        **env,
        "native library": str(native.lib_path()),
        "ICU": f"{icu_version} (Unicode {unicode_version})",
    }
    header = [
        "# Native sparse TF-IDF similarity vs the Python implementation (Phase 2)",
        "",
        "\n".join(f"- {k}: {v}" for k, v in env.items()),
        "",
        "Medians and p95 are over the timed runs (linear interpolation, numpy default), warmup runs excluded. "
        "Times are wall clock around the call; database calls go to a local throwaway Postgres. Native and "
        "Python rank the same documents in the same order; equality of their results is checked by the parity "
        "gate above each table and by tests/test_native_similarity.py (every top-10 list identical, scores "
        "within 1e-9), not assumed. The 'warm' rows are the production comparison: Python's cache-hit path "
        "against the native cache-hit path. 'Median ratio' is the Python median divided by the native median.",
        "",
        "Memory is the resident-set growth of a fresh process (corpus texts already in memory, library and ICU "
        "data already touched) between before and after building the index and a full garbage collection, "
        "measured with `ps -o rss`. The Python figure is the dict-of-dicts the cache holds; the native figure "
        "is the CSR index. RSS is page-granular and includes allocator slack, so small differences are noise.",
        "",
    ]
    return "\n".join(header) + "\n" + "\n\n".join(sections) + "\n"
