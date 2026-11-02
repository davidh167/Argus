"""Does Unicode tokenization matter for similar-roles results? (Decision D4 of the native engine plan.)

relevance.tokenize is `re.findall(r"\\w+", text.lower())`: Python's `\\w` is Unicode-aware, so "café"
is one token. A C++ port with an ASCII-only rule would see "caf" instead (and drop CJK entirely).
This scan counts how much of a corpus is affected and, for every posting, whether its top-10 similar
postings differ between the two tokenizers.

Similarity here is the same formula as relevance.py (tf = count/len, idf = log(N/(1+df)), cosine over
the tf-idf vectors, title repeated 3x) computed with scipy sparse matrices so the whole corpus can be
scanned, not just one posting. bench tests check it against relevance.find_similar_postings. Ties are
broken by corpus order after rounding similarities to 12 decimals, identically in both runs.
"""

import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import sparse

TOP_K = 10
_BATCH = 256

_UNICODE_TOKEN_RE = re.compile(r"\w+")
_ASCII_TOKEN_RE = re.compile(r"\w+", re.ASCII)


def tokenize_unicode(text: str) -> list[str]:
    return _UNICODE_TOKEN_RE.findall(text.lower())


def tokenize_ascii(text: str) -> list[str]:
    return _ASCII_TOKEN_RE.findall(text.lower())


def similarity_text(title: str, text: str) -> str:
    return (title + " ") * 3 + text


def _is_non_ascii(token: str) -> bool:
    return not token.isascii()


def tfidf_matrix(token_lists: list[list[str]]) -> sparse.csr_matrix:
    """Row-L2-normalized tf-idf matrix (rows = documents) using relevance.py's formulas."""
    n_docs = len(token_lists)
    vocab: dict[str, int] = {}
    counts_per_doc = []
    df: Counter[int] = Counter()
    for tokens in token_lists:
        counts = Counter(vocab.setdefault(t, len(vocab)) for t in tokens)
        counts_per_doc.append((counts, len(tokens)))
        df.update(counts.keys())
    idf = np.zeros(len(vocab))
    for term_id, count in df.items():
        idf[term_id] = math.log(n_docs / (1 + count))

    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for counts, length in counts_per_doc:
        for term_id, count in counts.items():
            indices.append(term_id)
            data.append(count / length * idf[term_id])
        indptr.append(len(indices))
    matrix = sparse.csr_matrix(
        (np.array(data), np.array(indices), np.array(indptr)), shape=(n_docs, len(vocab))
    )
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    norms[norms == 0.0] = (
        1.0  # an all-zero vector scores 0 against everything, like cosine_similarity
    )
    return sparse.diags(1.0 / norms) @ matrix  # type: ignore[return-value]


def top_k_neighbors(token_lists: list[list[str]], k: int = TOP_K) -> list[tuple[int, ...]]:
    """For each document, the row indices of its k most similar other documents, best first."""
    matrix = tfidf_matrix(token_lists)
    transposed = matrix.T.tocsc()
    n = matrix.shape[0]
    k = min(k, n - 1)
    out: list[tuple[int, ...]] = []
    for start in range(0, n, _BATCH):
        sims = np.round((matrix[start : start + _BATCH] @ transposed).toarray(), 12)
        rows = np.arange(sims.shape[0])
        sims[rows, start + rows] = -np.inf
        kth = np.partition(sims, -k, axis=1)[:, -k]
        for r in range(sims.shape[0]):
            # Everything tied with the k-th value is a candidate, so ties at the cut are resolved by
            # corpus order rather than by partition's arbitrary choice.
            candidates = np.flatnonzero(sims[r] >= kth[r])
            order = candidates[np.argsort(-sims[r, candidates], kind="stable")]
            out.append(tuple(int(i) for i in order[:k]))
    return out


@dataclass
class ScanResult:
    n_postings: int
    n_postings_non_ascii: int
    n_tokens: int
    n_tokens_non_ascii: int
    n_distinct_non_ascii_tokens: int
    n_top10_order_changed: int
    n_top10_set_changed: int
    n_top1_changed: int
    n_order_changed_with_non_ascii_text: int
    n_set_changed_with_non_ascii_text: int


def scan(documents: list[tuple[str, str]], progress: Callable[[str], None] = print) -> ScanResult:
    """documents: (title, text) per posting."""
    texts = [similarity_text(title, text) for title, text in documents]
    unicode_tokens = [tokenize_unicode(t) for t in texts]
    non_ascii_per_doc = [sum(map(_is_non_ascii, tokens)) for tokens in unicode_tokens]
    distinct = {t for tokens in unicode_tokens for t in tokens if _is_non_ascii(t)}

    progress("scan: unicode top-10")
    unicode_top = top_k_neighbors(unicode_tokens)
    progress("scan: ascii top-10")
    ascii_top = top_k_neighbors([tokenize_ascii(t) for t in texts])

    order_changed = [i for i, (a, b) in enumerate(zip(unicode_top, ascii_top)) if a != b]
    set_changed = [i for i, (a, b) in enumerate(zip(unicode_top, ascii_top)) if set(a) != set(b)]
    return ScanResult(
        n_postings=len(documents),
        n_postings_non_ascii=sum(1 for n in non_ascii_per_doc if n),
        n_tokens=sum(map(len, unicode_tokens)),
        n_tokens_non_ascii=sum(non_ascii_per_doc),
        n_distinct_non_ascii_tokens=len(distinct),
        n_top10_order_changed=len(order_changed),
        n_top10_set_changed=len(set_changed),
        n_top1_changed=sum(1 for a, b in zip(unicode_top, ascii_top) if a[0] != b[0]),
        n_order_changed_with_non_ascii_text=sum(1 for i in order_changed if non_ascii_per_doc[i]),
        n_set_changed_with_non_ascii_text=sum(1 for i in set_changed if non_ascii_per_doc[i]),
    )
