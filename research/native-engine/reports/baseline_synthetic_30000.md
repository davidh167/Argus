# Python baseline for the native engine (Phase 0)

- generated: 2026-11-02 08:23 UTC
- machine: macOS-27.0.1-arm64-arm-64bit, arm64, 10 logical CPUs
- python: 3.12.2
- libraries: numpy 2.5.3, scipy 1.18.1, bm25s 0.3.14, sqlalchemy 2.0.52
- postgres: PostgreSQL 17.11
- git commit: 3e7d998

Medians and p95 are over the timed runs (p95: linear interpolation between order statistics, numpy default). Each case runs its warmup runs first, untimed. Times are wall clock around the call; every database call goes to a local throwaway Postgres.

## Corpus: synthetic-30000 (N = 30,000)

- seed: 20261101
- non-ASCII posting fraction: 0.06
- rescore_all rescored: 30000
- market_demand clusters: 8289
- timed runs: 10 per case (fewer than 30: the cases are too slow at this N)

### Timings

| Case | Median ms | p95 ms | Min ms | Max ms | Timed runs | Warmup |
|---|---:|---:|---:|---:|---:|---:|
| find_similar_postings (cold) | 11,379 | 13,702 | 9,769 | 15,140 | 10 | 1 |
| find_similar_postings (warm) | 1,181 | 1,333 | 889 | 1,354 | 10 | 1 |
| rescore_all | 22,467 | 37,178 | 21,340 | 48,697 | 10 | 1 |
| bm25_scores | 6,256 | 10,343 | 5,420 | 11,535 | 10 | 1 |
| market_demand (stub embedder) | 68,333 | 73,583 | 58,048 | 73,749 | 10 | 1 |
|   component: DB fetch | 1,718 | 2,053 | 1,146 | 2,078 | 10 | 1 |
|   component: tokenize | 7,357 | 13,367 | 3,814 | 15,788 | 10 | 1 |
|   component: df/idf + tf-idf vectors | 4,898 | 5,597 | 3,728 | 5,677 | 10 | 1 |
|   component: cosine scan + top-10 | 1,298 | 1,332 | 1,231 | 1,344 | 10 | 1 |

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
| warm find_similar p95 | 1,333 ms | < 50 ms | no |
| cold find_similar p95 | 13,702 ms | < 1000 ms | no |

Kill criterion (plan section 3, Phase 0) satisfied: no.

### Non-ASCII tokenization scan (decision D4)

| Measure | Value |
|---|---:|
| postings | 30,000 |
| postings with at least one non-ASCII word token | 1,985 (6.62%) |
| word tokens (Unicode `\w+`) | 13,732,103 |
| tokens containing a non-ASCII word character | 4,023 (0.03%) |
| distinct such tokens | 22 |
| postings whose top-10 similar list (order) differs, ASCII-only vs Unicode | 4,770 (15.90%) |
| postings whose top-10 similar set differs | 1,443 (4.81%) |
| postings whose top-1 similar posting differs | 164 (0.55%) |
| of the order-changed postings, own text contains non-ASCII tokens | 418 |
| of the set-changed postings, own text contains non-ASCII tokens | 139 |
