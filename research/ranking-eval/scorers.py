"""Step 2: compute one score per posting for every system under test.

Every scorer here is QUERY-INDEPENDENT with respect to the evaluation
queries (it scores a posting against the user's profile, not against the
held-out positive), so the same scores are reused across all leave-one-out
queries. Writes data/scores.npz.

Systems
  random        seeded noise -- floor
  stage1        free title-regex tier (fast_track > analyze > skip)
  tfidf         the app's persisted TF-IDF cosine relevance_score
  bm25          bm25s lexical match of the profile query against title+JD
  embed         nomic-embed-text cosine (query/document task prefixes) of
                profile query vs title + start of the JD
  bm25_app      the app's own BM25 (app.services.relevance), parity check
  rrf           reciprocal-rank fusion of bm25 + embed (ranx)
Leakage canaries (should score ~chance; if they don't, the label is
detectable from artifacts and every other number is suspect)
  canary_length   document length
  canary_format   fraction of non-alphanumeric/whitespace characters

Deliberately NOT included: any Stage 2/3 LLM verdict. Stage 3 exists only for
postings the user chose to analyze (the positives), so it would be pure
label leakage; Stage 2 is skipped for fast-tracked postings, so its coverage
differs by class. See README.

Run:  PYTHONPATH=. poetry run python research/ranking-eval/scorers.py
"""

import json
import re
from pathlib import Path

import bm25s
import numpy as np
from ranx import Run, fuse

from app.services import embeddings

HERE = Path(__file__).parent
DATA = HERE / "data"
SEED = 7
DOC_CHARS = 1200  # nomic-embed-text handles far more; this keeps 3k docs fast


def load():
    rows = [json.loads(line) for line in open(DATA / "snapshot.jsonl")]
    meta = json.loads((DATA / "meta.json").read_text())
    return rows, meta


def rrf(score_sets: dict[str, np.ndarray], ids: list[int]) -> np.ndarray:
    runs = [
        Run({"q": {str(i): float(s) for i, s in zip(ids, scores)}}, name=name)
        for name, scores in score_sets.items()
    ]
    fused = fuse(runs, method="rrf")
    out = fused.run["q"]
    return np.array([out[str(i)] for i in ids])


def main() -> None:
    rows, meta = load()
    ids = [r["id"] for r in rows]
    n = len(rows)
    rng = np.random.default_rng(SEED)
    scores: dict[str, np.ndarray] = {}

    scores["random"] = rng.random(n)
    scores["stage1"] = np.array(
        [{"fast_track": 2, "analyze": 1, "skip": 0}[r["stage1"]] for r in rows], dtype=float
    )
    scores["tfidf"] = np.array([r["relevance_score"] or 0.0 for r in rows])

    docs = [f"{r['title']} {r['text']}" for r in rows]
    corpus_tokens = bm25s.tokenize(docs, stopwords="en", show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    query_tokens = bm25s.tokenize(
        meta["profile_query"], stopwords="en", show_progress=False, return_ids=False
    )[0]
    # get_scores wants tokens the index knows; unknown ones are ignored.
    scores["bm25"] = np.asarray(retriever.get_scores(query_tokens), dtype=float)

    # The shipped function (app.services.relevance.bm25_scores). It is the same bm25s
    # configuration as "bm25" above, so these two must agree: a regression check.
    from app.services import relevance

    scores["bm25_app"] = np.array(relevance.bm25_scores(docs, meta["profile_query"]))

    doc_texts = [f"search_document: {r['title']}\n{r['text'][:DOC_CHARS]}" for r in rows]
    doc_vecs = embeddings.embed_texts(doc_texts)
    query_vec = embeddings.embed_texts([f"search_query: {meta['profile_query']}"])[0]
    scores["embed"] = doc_vecs @ query_vec

    scores["rrf"] = rrf({"bm25": scores["bm25"], "embed": scores["embed"]}, ids)

    scores["canary_length"] = np.array([len(r["text"]) for r in rows], dtype=float)
    scores["canary_format"] = np.array(
        [len(re.findall(r"[^\w\s]", r["text"])) / max(len(r["text"]), 1) for r in rows]
    )

    np.savez(DATA / "scores.npz", ids=np.array(ids), **scores)
    print("wrote", DATA / "scores.npz", {k: round(float(v.mean()), 4) for k, v in scores.items()})


if __name__ == "__main__":
    main()
