"""The timed cases: real entry points in app/services, called the way the app calls them.

Every case is `warmup` untimed runs, then `runs` timed runs (time.perf_counter around the call only;
`before` hooks run untimed). Statistics are computed from the raw samples in bench_report.
"""

import random
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.posting import Posting
from app.schemas.profile import ProfileSnapshotData
from app.services import profile as profile_service
from app.services import relevance, roadmap
from bench_report import COLD_CASE, WARM_CASE, CaseResults, Timing
from bench_stub import stubbed_embedder

SessionFactory = Callable[[], Session]


@contextmanager
def similarity_engine(name: str) -> Iterator[None]:
    """Runs the block with settings.similarity_engine set to name (the Phase 0 cases pin "python")."""
    previous = settings.similarity_engine
    settings.similarity_engine = name  # type: ignore[assignment]
    try:
        yield
    finally:
        settings.similarity_engine = previous


def measure(
    name: str,
    what: str,
    fn: Callable[[], Any],
    *,
    runs: int,
    warmup: int,
    before: Callable[[], None] | None = None,
) -> tuple[Timing, Any]:
    """Returns the Timing and the last run's result."""
    result = None
    samples: list[float] = []
    for i in range(warmup + runs):
        if before is not None:
            before()
        start = time.perf_counter()
        result = fn()
        elapsed_ms = (time.perf_counter() - start) * 1000
        if i >= warmup:
            samples.append(elapsed_ms)
    print(
        f"  {name.strip()}: {sorted(samples)[len(samples) // 2]:.1f} ms median over {len(samples)} runs",
        file=sys.stderr,
        flush=True,
    )
    return Timing(name, what, samples, warmup), result


def target_posting_ids(
    session_factory: SessionFactory, count: int = 50, seed: int = 1
) -> list[int]:
    """A fixed seeded sample of posting ids; find_similar cases cycle through them so one lucky or
    unlucky posting does not set the number."""
    with session_factory() as db:
        ids = list(db.scalars(select(Posting.id).order_by(Posting.id)))
    return random.Random(seed).sample(ids, min(count, len(ids)))


def run_cases(session_factory: SessionFactory, *, runs: int, warmup: int) -> CaseResults:
    with similarity_engine("python"):
        return _run_cases(session_factory, runs=runs, warmup=warmup)


def _run_cases(session_factory: SessionFactory, *, runs: int, warmup: int) -> CaseResults:
    out = CaseResults()
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

    def clear_cache() -> None:
        relevance._similarity_index_cache.clear()

    t, _ = measure(
        COLD_CASE,
        "similarity cache cleared before each call: DB fetch + tokenize + tf-idf vectors + cosine scan",
        find_similar,
        runs=runs,
        warmup=warmup,
        before=clear_cache,
    )
    out.timings.append(t)

    clear_cache()
    find_similar()  # fills the cache; the warmup runs below keep it warm
    t, _ = measure(
        WARM_CASE,
        "cache hit: cache-key query + cosine scan of every vector + top-10 fetch",
        find_similar,
        runs=runs,
        warmup=warmup,
    )
    out.timings.append(t)

    def rescore() -> dict:
        with session_factory() as db:
            result = relevance.rescore_all(db)
            db.rollback()  # flush happened inside the call; leave the table as it was
        return result

    t, result = measure(
        "rescore_all",
        "tokenize + df/idf + bm25 + per-posting tf-idf vector and matched terms + ORM flush "
        "(rolled back, rollback not timed beyond the call)",
        rescore,
        runs=runs,
        warmup=warmup,
    )
    out.timings.append(t)
    out.facts["rescore_all rescored"] = str(result["rescored"])

    with session_factory() as db:
        rows = list(
            db.execute(
                select(Posting.id, Posting.title, Posting.normalized_jd_text, Posting.company_id)
            )
        )
        profile = ProfileSnapshotData.model_validate(
            profile_service.get_current_profile(db).snapshot
        )
    query_text = relevance.profile_query_text(profile)
    bm25_texts = [f"{title} {normalized or ''}" for _id, title, normalized, _cid in rows]
    t, _ = measure(
        "bm25_scores",
        "bm25s tokenize + index + score of the profile query against every posting (as rescore_all)",
        lambda: relevance.bm25_scores(bm25_texts, query_text),
        runs=runs,
        warmup=warmup,
    )
    out.timings.append(t)

    with stubbed_embedder():

        def market_demand() -> list:
            with session_factory() as db:
                return roadmap.market_demand(db)

        t, clusters = measure(
            "market_demand (stub embedder)",
            "ORM load of postings + verdicts, phrase normalization, stub embeddings, average-link + "
            "greedy clustering, classification",
            market_demand,
            runs=runs,
            warmup=warmup,
        )
    out.timings.append(t)
    out.facts["market_demand clusters"] = str(len(clusters))

    out.timings.extend(
        _find_similar_components(session_factory, rows, targets[0], runs=runs, warmup=warmup)
    )
    return out


def _find_similar_components(
    session_factory: SessionFactory, rows: list, target_id: int, *, runs: int, warmup: int
) -> list[Timing]:
    """The cold find_similar cost split into its steps, re-composed from the same relevance
    functions _get_similarity_index and find_similar_postings call (not the entry point itself)."""

    def fetch() -> list:
        with session_factory() as db:
            return list(
                db.execute(
                    select(
                        Posting.id, Posting.title, Posting.normalized_jd_text, Posting.company_id
                    )
                )
            )

    t_fetch, _ = measure(
        "  component: DB fetch",
        "select id, title, normalized_jd_text, company_id for every posting",
        fetch,
        runs=runs,
        warmup=warmup,
    )
    texts = [relevance._similarity_text(title, normalized) for _id, title, normalized, _c in rows]
    t_tokenize, token_lists = measure(
        "  component: tokenize",
        "relevance.tokenize over every posting's title x3 + text",
        lambda: [relevance.tokenize(t) for t in texts],
        runs=runs,
        warmup=warmup,
    )

    def build_vectors() -> dict[int, dict[str, float]]:
        df = relevance.document_frequencies(token_lists)
        idf = relevance.inverse_document_frequencies(df, n_docs=len(rows))
        return {
            row[0]: relevance.tfidf_vector(relevance.term_frequencies(tokens), idf)
            for row, tokens in zip(rows, token_lists)
        }

    t_vectors, vectors = measure(
        "  component: df/idf + tf-idf vectors",
        "document_frequencies, inverse_document_frequencies, tfidf_vector for every posting",
        build_vectors,
        runs=runs,
        warmup=warmup,
    )
    target_vec = vectors[target_id]

    def scan() -> list:
        return sorted(
            (
                (pid, relevance.cosine_similarity(target_vec, vec))
                for pid, vec in vectors.items()
                if pid != target_id
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )[:10]

    t_scan, _ = measure(
        "  component: cosine scan + top-10",
        "cosine_similarity of one target against every vector, sorted",
        scan,
        runs=runs,
        warmup=warmup,
    )
    return [t_fetch, t_tokenize, t_vectors, t_scan]
