# Python baseline for the native engine (Phase 0)

- generated: 2026-11-02 07:55 UTC
- machine: macOS-27.0.1-arm64-arm-64bit, arm64, 10 logical CPUs
- python: 3.12.2
- libraries: numpy 2.5.3, scipy 1.18.1, bm25s 0.3.14, sqlalchemy 2.0.52
- postgres: PostgreSQL 17.11
- git commit: 3e7d998

Medians and p95 are over the timed runs (p95: linear interpolation between order statistics, numpy default). Each case runs its warmup runs first, untimed. Times are wall clock around the call; every database call goes to a local throwaway Postgres.

## Corpus: synthetic-3000 (N = 3,000)

- seed: 20261101
- non-ASCII posting fraction: 0.06
- rescore_all rescored: 3000
- market_demand clusters: 2576

### Timings

| Case | Median ms | p95 ms | Min ms | Max ms | Timed runs | Warmup |
|---|---:|---:|---:|---:|---:|---:|
| find_similar_postings (cold) | 651 | 1,029 | 601 | 1,282 | 30 | 3 |
| find_similar_postings (warm) | 66.8 | 87.3 | 53.6 | 111 | 30 | 3 |
| rescore_all | 1,562 | 2,227 | 1,311 | 2,357 | 30 | 3 |
| bm25_scores | 507 | 729 | 491 | 1,185 | 30 | 3 |
| market_demand (stub embedder) | 4,188 | 4,788 | 3,325 | 5,319 | 30 | 3 |
|   component: DB fetch | 109 | 226 | 91.1 | 256 | 30 | 3 |
|   component: tokenize | 261 | 387 | 234 | 436 | 30 | 3 |
|   component: df/idf + tf-idf vectors | 288 | 363 | 256 | 401 | 30 | 3 |
|   component: cosine scan + top-10 | 66.5 | 118 | 60.0 | 132 | 30 | 3 |

Case definitions:

- find_similar_postings (cold): similarity cache cleared before each call: DB fetch + tokenize + tf-idf vectors + cosine scan
- find_similar_postings (warm): cache hit: cache-key query + cosine scan of every vector + top-10 fetch
- rescore_all: tokenize + df/idf + bm25 + per-posting tf-idf vector and matched terms + ORM flush (rolled back, rollback not timed beyond the call)
- bm25_scores: bm25s tokenize + index + score of the profile query against every posting (as rescore_all)
- market_demand (stub embedder): ORM load of postings + verdicts, phrase normalization, stub embeddings, average-link + greedy clustering, classification
- component: DB fetch: select id, title, normalized_jd_text, company_id for every posting
- component: tokenize: relevance.tokenize over every posting's title x3 + text
- component: df/idf + tf-idf vectors: document_frequencies, inverse_document_frequencies, tfidf_vector for every posting
- component: cosine scan + top-10: cosine_similarity of one target against every vector, sorted

### Phase 0 kill criterion

| Condition | Measured | Limit | Holds |
|---|---:|---:|---|
| warm find_similar p95 | 87.3 ms | < 50 ms | no |
| cold find_similar p95 | 1,029 ms | < 1000 ms | no |

Kill criterion (plan section 3, Phase 0) satisfied: no.

### Non-ASCII tokenization scan (decision D4)

| Measure | Value |
|---|---:|
| postings | 3,000 |
| postings with at least one non-ASCII word token | 181 (6.03%) |
| word tokens (Unicode `\w+`) | 1,380,964 |
| tokens containing a non-ASCII word character | 348 (0.03%) |
| distinct such tokens | 22 |
| postings whose top-10 similar list (order) differs, ASCII-only vs Unicode | 312 (10.40%) |
| postings whose top-10 similar set differs | 94 (3.13%) |
| postings whose top-1 similar posting differs | 6 (0.20%) |
| of the order-changed postings, own text contains non-ASCII tokens | 20 |
| of the set-changed postings, own text contains non-ASCII tokens | 5 |
