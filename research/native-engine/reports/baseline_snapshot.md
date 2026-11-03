# Python baseline for the native engine (Phase 0)

- generated: 2026-11-02 17:41 UTC
- machine: macOS-27.0.1-arm64-arm-64bit, arm64, 10 logical CPUs
- python: 3.12.2
- libraries: numpy 2.5.3, scipy 1.18.1, bm25s 0.3.14, sqlalchemy 2.0.52
- postgres: PostgreSQL 17.11
- git commit: e20f39f

Medians and p95 are over the timed runs (p95: linear interpolation between order statistics, numpy default). Each case runs its warmup runs first, untimed. Times are wall clock around the call; every database call goes to a local throwaway Postgres.

## Corpus: snapshot-3402 (N = 3,402)

- exported_at: 2026-11-02T17:36:55.649522+00:00
- key_skills: synthesized (the snapshot has no stage2_key_skills)
- rescore_all rescored: 3402
- market_demand clusters: 2860

### Timings

| Case | Median ms | p95 ms | Min ms | Max ms | Timed runs | Warmup |
|---|---:|---:|---:|---:|---:|---:|
| find_similar_postings (cold) | 782 | 889 | 725 | 919 | 30 | 3 |
| find_similar_postings (warm) | 55.9 | 153 | 46.4 | 162 | 30 | 3 |
| rescore_all | 1,625 | 1,757 | 1,572 | 1,794 | 30 | 3 |
| bm25_scores | 629 | 676 | 613 | 679 | 30 | 3 |
| market_demand (stub embedder) | 3,297 | 4,550 | 2,675 | 4,716 | 30 | 3 |
|   component: DB fetch | 68.2 | 111 | 61.9 | 116 | 30 | 3 |
|   component: tokenize | 279 | 297 | 270 | 354 | 30 | 3 |
|   component: df/idf + tf-idf vectors | 322 | 337 | 320 | 341 | 30 | 3 |
|   component: cosine scan + top-10 | 139 | 145 | 136 | 187 | 30 | 3 |

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
| warm find_similar p95 | 153 ms | < 50 ms | no |
| cold find_similar p95 | 889 ms | < 1000 ms | yes |

Kill criterion (plan section 3, Phase 0) satisfied: no.

### Non-ASCII tokenization scan (decision D4)

| Measure | Value |
|---|---:|
| postings | 3,402 |
| postings with at least one non-ASCII word token | 35 (1.03%) |
| word tokens (Unicode `\w+`) | 1,738,790 |
| tokens containing a non-ASCII word character | 559 (0.03%) |
| distinct such tokens | 259 |
| postings whose top-10 similar list (order) differs, ASCII-only vs Unicode | 122 (3.59%) |
| postings whose top-10 similar set differs | 45 (1.32%) |
| postings whose top-1 similar posting differs | 6 (0.18%) |
| of the order-changed postings, own text contains non-ASCII tokens | 10 |
| of the set-changed postings, own text contains non-ASCII tokens | 6 |
