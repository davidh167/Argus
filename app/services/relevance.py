"""Classical-NLP (TF-IDF + cosine similarity) relevance ranking, used to cap
LLM spend before Stage 2 batch submission -- see stage2_batch.submit_pending_batch.
Deliberately dependency-free (stdlib only): at the corpus sizes this ranks
(hundreds of postings, not millions), hand-rolled sparse dict vectors are
simpler, fully inspectable, and just as fast as pulling in numpy/scipy via
scikit-learn for what is here a single query-vector-against-N-docs pass.
"""

import math
import re
from collections import Counter
from typing import Any

import bm25s
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.posting import Posting
from app.schemas.profile import ProfileSnapshotData
from app.services import profile as profile_service

# Above this cosine similarity, two postings at the same company are flagged "possibly the same role
# on another team or requisition". Informational only: different external ids are always different
# roles, so this is never a merge or dedup signal.
#
# Calibrated against two real same-company samples (300 postings each) that disagree: one company's
# non-duplicate pairs top out at 0.70 with true duplicates at ~1.0, while another's postings share
# long team-level prose, so even a 0.99 cutoff flags related-but-distinct roles. No single threshold
# separates both; 0.9 is set from the first and is a known limitation for the second.
SIMILAR_ROLE_SAME_COMPANY_THRESHOLD = 0.9

# In-process cache of the corpus's similarity vectors: rebuilding them dominates find_similar_postings
# (~0.5 s of ~0.8 s at ~2,700 postings) and it runs on every posting detail view. Keyed on (row count,
# max last_seen_at, max detail_fetched_at), which together change whenever postings are upserted or
# hydrated.
_similarity_index_cache: dict[str, tuple] = {}

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def term_frequencies(tokens: list[str]) -> dict[str, float]:
    """Raw counts normalized by document length so longer JDs don't win on
    volume alone."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}


def document_frequencies(token_lists: list[list[str]]) -> dict[str, int]:
    df: dict[str, int] = {}
    for tokens in token_lists:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    return df


def inverse_document_frequencies(df: dict[str, int], n_docs: int) -> dict[str, float]:
    """+1 smoothing keeps this finite even for a term present in every doc."""
    return {term: math.log(n_docs / (1 + count)) for term, count in df.items()}


def tfidf_vector(tf: dict[str, float], idf: dict[str, float]) -> dict[str, float]:
    """Terms absent from idf's vocabulary (e.g. a query term no candidate
    document contains at all) are dropped -- they can't contribute to any
    dot product against a corpus doc, so keeping them is pure waste."""
    return {term: weight * idf[term] for term, weight in tf.items() if term in idf}


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    shared = a.keys() & b.keys()
    dot = sum(a[t] * b[t] for t in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def bm25_scores(texts: list[str], query_text: str) -> list[float]:
    """BM25 (bm25s library, Lucene variant, k1=1.5, b=0.75, English stopwords
    dropped) of query_text against each text. This is the configuration the ranking
    eval validated (research/ranking-eval): it beat the TF-IDF cosine and the
    Stage 1 title tier on held-out manual picks. bm25s's default tokenizer drops
    one-character tokens (so "C" and "R" never match). Scores are unbounded and only
    meaningful relative to this corpus."""
    if not texts:
        return []
    query = bm25s.tokenize(query_text, stopwords="en", show_progress=False, return_ids=False)[0]
    if not query:  # bm25s raises on an empty query; an empty profile query just matches nothing
        return [0.0] * len(texts)
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(texts, stopwords="en", show_progress=False), show_progress=False)
    return retriever.get_scores(query).tolist()


def rank_by_relevance(items: list[tuple[Any, str]], query_text: str) -> list[Any]:
    """items: (item, text_to_score) pairs, e.g. (posting, posting.raw_jd_text).
    Returns the items (not the pairs) reordered most-relevant-first by BM25
    of query_text against each item's text. Statistics (idf, average
    length) come from the candidate items alone; query_text is not folded
    into the corpus."""
    if not items:
        return []
    scores = bm25_scores([text for _, text in items], query_text)
    order = sorted(
        range(len(items)), key=lambda i: scores[i], reverse=True
    )  # stable: ties keep input order
    return [items[i][0] for i in order]


def top_matched_terms(
    doc_vec: dict[str, float], query_vec: dict[str, float], n: int = 8
) -> list[str]:
    """The n terms contributing most to doc_vec/query_vec's cosine
    similarity dot product, highest contribution first -- cheap "why is
    this ranked here" explainability. A term only counts if it's present
    in BOTH vectors (shared terms are exactly what the dot product sums
    over); ranked by each term's product, not by either vector's weight
    alone, since a term with a huge document weight but near-zero query
    weight barely moved the score."""
    shared = doc_vec.keys() & query_vec.keys()
    ranked = sorted(shared, key=lambda t: doc_vec[t] * query_vec[t], reverse=True)
    return ranked[:n]


def rescore_all(db: Session) -> dict:
    """Recomputes relevance_score + relevance_matched_terms for EVERY
    posting in one pass against the CURRENT profile, over normalized_jd_text
    (not raw_jd_text, which still carries same-company EEO/benefits
    boilerplate -- see jd_sections.build_normalized_jd_text). One pass, not
    one query per posting, because IDF isn't well-defined for a single
    document -- it needs the whole corpus. Call after a crawl/scrape/
    hydrate brings in new postings, and after a profile edit (both change
    what "relevant" means); see callers for exactly where.

    Services flush, don't commit -- matches this codebase's convention;
    the caller commits.
    """
    profile_version = profile_service.get_current_profile(db)
    profile = ProfileSnapshotData.model_validate(profile_version.snapshot)
    query_text = profile_query_text(profile)

    postings = list(db.scalars(select(Posting)))
    if not postings:
        return {"rescored": 0, "profile_version_id": profile_version.id}

    texts = [f"{p.title} {p.normalized_jd_text or ''}" for p in postings]
    token_lists = [tokenize(t) for t in texts]
    df = document_frequencies(token_lists)
    idf = inverse_document_frequencies(df, n_docs=len(postings))
    query_tokens = tokenize(query_text)
    query_vec = tfidf_vector(term_frequencies(query_tokens), idf)
    scores = bm25_scores(texts, query_text)

    # relevance_score is BM25; matched terms stay tf-idf contributions,
    # which is only used to explain the match, not to rank.
    for posting, tokens, score in zip(postings, token_lists, scores):
        doc_vec = tfidf_vector(term_frequencies(tokens), idf)
        posting.relevance_score = score
        posting.relevance_matched_terms = top_matched_terms(doc_vec, query_vec)
        posting.relevance_profile_version_id = profile_version.id

    db.flush()
    return {"rescored": len(postings), "profile_version_id": profile_version.id}


def profile_query_text(profile: ProfileSnapshotData) -> str:
    """The pseudo-document JD text is scored against. Positive-signal
    fields only: target_signals + skills.core + skills.developing.
    skip_patterns is deliberately excluded -- it's a negative signal, and
    cosine similarity can't express "penalize overlap with this": a JD that
    happens to share vocabulary with a skip pattern would score HIGHER, not
    lower, which is backwards. Stage 1's title rules and Stage 2's own LLM
    judgment already handle exclusion; this function only ranks, it doesn't
    filter."""
    parts = profile.criteria.target_signals + profile.skills.core + profile.skills.developing
    return " ".join(parts)


def _similarity_text(title: str, normalized_jd_text: str | None) -> str:
    """Title repeated 3x + normalized_jd_text -- a short title's vocabulary
    would otherwise be swamped by a much longer JD in a plain TF-IDF
    comparison, so it's weighted up explicitly rather than left to chance.
    normalized_jd_text (not raw_jd_text), same reasoning as rescore_all:
    same-company EEO/benefits boilerplate would otherwise dominate.
    Takes plain strings, not a Posting, so callers building this from a
    raw SQL row (see _get_similarity_index, which avoids loading ORM
    instances into its cross-request cache) don't need one."""
    return (title + " ") * 3 + (normalized_jd_text or "")


def _similarity_cache_key(db: Session):
    return db.execute(
        select(
            func.count(), func.max(Posting.last_seen_at), func.max(Posting.detail_fetched_at)
        ).select_from(Posting)
    ).one()


def _get_similarity_index(
    db: Session,
) -> tuple[dict[int, dict[str, float]], dict[int, int]]:
    """({posting_id: tfidf_vector}, {posting_id: company_id}) for the WHOLE
    corpus -- cached, see _similarity_index_cache's module-level
    docstring. Deliberately caches only plain ints/dicts, never ORM
    Posting instances: those are bound to the Session that loaded them,
    and a FastAPI request's Session is closed once the request ends
    (Depends(get_db)) -- a cached ORM object reused on a LATER request
    raises DetachedInstanceError the moment anything touches a
    lazy-loaded attribute (e.g. posting.company in _to_summary). Found
    this live: the first call after a cache miss worked, the second
    (a fresh request/session reusing the cached objects) 500'd.
    """
    key = _similarity_cache_key(db)
    cached = _similarity_index_cache.get("index")
    if cached is not None and cached[0] == key:
        return cached[1]

    rows = list(
        db.execute(
            select(Posting.id, Posting.title, Posting.normalized_jd_text, Posting.company_id)
        )
    )
    token_lists = [
        tokenize(_similarity_text(title, normalized)) for _id, title, normalized, _cid in rows
    ]
    df = document_frequencies(token_lists)
    idf = inverse_document_frequencies(df, n_docs=len(rows) or 1)
    vectors = {
        pid: tfidf_vector(term_frequencies(tokens), idf)
        for (pid, _title, _normalized, _cid), tokens in zip(rows, token_lists)
    }
    company_ids = {pid: cid for pid, _title, _normalized, cid in rows}

    result = (vectors, company_ids)
    _similarity_index_cache["index"] = (key, result)
    return result


def find_similar_postings(
    db: Session, posting_id: int, limit: int = 10
) -> list[tuple[Posting, float, bool]] | None:
    """Cosine similarity over normalized_jd_text (+ title, weighted -- see
    _similarity_text) across the whole corpus. Returns None when
    posting_id doesn't exist (the caller 404s). Purely informational
    ranking -- different external_ids are always different roles, and this
    never merges or dedupes anything.

    Returns (posting, similarity, possibly_same_role) tuples, highest
    similarity first. possibly_same_role is True only for a same-company
    pair above SIMILAR_ROLE_SAME_COMPANY_THRESHOLD -- see that constant
    for how it was calibrated against real data.
    """
    vectors, company_ids = _get_similarity_index(db)
    if posting_id not in vectors:
        return None
    target_vec = vectors[posting_id]
    target_company_id = company_ids[posting_id]

    scored = sorted(
        (
            (pid, cosine_similarity(target_vec, vec))
            for pid, vec in vectors.items()
            if pid != posting_id
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )[:limit]

    # Fresh ORM objects from THIS request's session, in the order just
    # computed -- never the cached (id, vector) data itself.
    top_ids = [pid for pid, _sim in scored]
    postings_by_id = {p.id: p for p in db.scalars(select(Posting).where(Posting.id.in_(top_ids)))}

    return [
        (
            postings_by_id[pid],
            sim,
            company_ids[pid] == target_company_id and sim >= SIMILAR_ROLE_SAME_COMPANY_THRESHOLD,
        )
        for pid, sim in scored
    ]
