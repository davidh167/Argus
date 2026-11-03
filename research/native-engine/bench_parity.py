"""Exact-parity check of the native similarity engine against the Python implementation.

For each target document: the Python reference (relevance.python_similarity_vectors + python_top_k, i.e. the
code find_similar_postings runs under SIMILARITY_ENGINE=python) and the native engine (native.Similarity)
rank the same corpus, fed in the same order. A target passes when both return the same ids in the same
order and every score is within SCORE_TOLERANCE. Used by tests/test_native_similarity.py and by
parity_snapshot.py (the real-corpus run).
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from app import native
from app.services import relevance

SCORE_TOLERANCE = 1e-9


@dataclass
class Mismatch:
    posting_id: int
    kind: str  # "ids": different ids or order; "score": same ids, a score off by more than the tolerance
    python: list[tuple[int, float]]
    native: list[tuple[int, float]]

    @property
    def scores_equivalent(self) -> bool:
        """The ids differ but the score at every rank is within tolerance: a reordering of near-equal
        scores, not a different ranking."""
        return len(self.python) == len(self.native) and all(
            abs(a - b) <= SCORE_TOLERANCE for (_, a), (_, b) in zip(self.python, self.native)
        )


@dataclass
class ParityResult:
    n_docs: int
    n_targets: int
    k: int
    max_abs_score_diff: float = 0.0
    n_exact_score_matches: int = 0  # hits whose scores are bit-identical
    n_hits: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)

    @property
    def n_id_mismatch(self) -> int:
        return sum(1 for m in self.mismatches if m.kind == "ids")

    @property
    def n_near_tie_reorders(self) -> int:
        return sum(1 for m in self.mismatches if m.kind == "ids" and m.scores_equivalent)

    @property
    def n_score_mismatch(self) -> int:
        return sum(1 for m in self.mismatches if m.kind == "score")

    @property
    def ok(self) -> bool:
        return not self.mismatches


def compare(
    documents: list[tuple[int, str]],
    *,
    k: int = 10,
    targets: list[int] | None = None,
    sample: int | None = None,
    seed: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> ParityResult:
    """documents: (id, text) in the order find_similar_postings feeds them. targets, or a seeded random
    `sample` of the ids, limits which documents are used as the query; default is every document."""
    ids = [pid for pid, _ in documents]
    if targets is None:
        targets = (
            ids if sample is None or sample >= len(ids) else random.Random(seed).sample(ids, sample)
        )
    vectors = relevance.python_similarity_vectors(documents)
    index = native.Similarity(documents)
    result = ParityResult(n_docs=len(documents), n_targets=len(targets), k=k)
    for done, target in enumerate(targets, 1):
        if progress is not None:
            progress(done, len(targets))
        expected = relevance.python_top_k(vectors, target, k)
        actual = index.top_k(target, k)
        assert actual is not None
        result.n_hits += len(expected)
        if [i for i, _ in expected] != [i for i, _ in actual] or len(expected) != len(actual):
            result.mismatches.append(Mismatch(target, "ids", expected, actual))
            continue
        diffs = [abs(a - b) for (_, a), (_, b) in zip(expected, actual)]
        result.n_exact_score_matches += sum(1 for d in diffs if d == 0.0)
        if diffs:
            result.max_abs_score_diff = max(result.max_abs_score_diff, max(diffs))
            if max(diffs) > SCORE_TOLERANCE:
                result.mismatches.append(Mismatch(target, "score", expected, actual))
    return result
