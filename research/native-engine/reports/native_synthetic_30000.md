# Native sparse TF-IDF similarity vs the Python implementation (Phase 2)

- generated: 2026-11-03 09:53 UTC
- machine: macOS-27.0.1-arm64-arm-64bit, arm64, 10 logical CPUs
- python: 3.12.2
- libraries: numpy 2.5.3, scipy 1.18.1, bm25s 0.3.14, sqlalchemy 2.0.52
- postgres: PostgreSQL 17.11
- git commit: e20f39f
- native library: /Users/dhern162/argus/.claude/worktrees/agent-a59589817fdbb857a/native/build-release/libargus.dylib
- ICU: 78.3 (Unicode 17.0)

Medians and p95 are over the timed runs (linear interpolation, numpy default), warmup runs excluded. Times are wall clock around the call; database calls go to a local throwaway Postgres. Native and Python rank the same documents in the same order; equality of their results is checked by the parity gate above each table and by tests/test_native_similarity.py (every top-10 list identical, scores within 1e-9), not assumed. The 'warm' rows are the production comparison: Python's cache-hit path against the native cache-hit path. 'Median ratio' is the Python median divided by the native median.

Memory is the resident-set growth of a fresh process (corpus texts already in memory, library and ICU data already touched) between before and after building the index and a full garbage collection, measured with `ps -o rss`. The Python figure is the dict-of-dicts the cache holds; the native figure is the CSR index. RSS is page-granular and includes allocator slack, so small differences are noise.

## Corpus: synthetic-30000 (N = 30,000)

- seed: 20261101
- non-ASCII posting fraction: 0.06
- parity gate (run before timing): 100 query postings (seeded sample) x top-10 against the Python implementation fed in the same order: 0 id/order mismatches, 0 score mismatches (tolerance 1e-09), max |score diff| 0, 1000 of 1000 scores bit-identical
- fewest timed runs in any case: 10

### Native vs Python

| What | Python median ms | Native median ms | Median ratio | Python p95 ms | Native p95 ms |
|---|---:|---:|---:|---:|---:|
| find_similar, cold (DB fetch + build + query) | 6,092 | 1,400 | 4.4x | 6,214 | 1,470 |
| find_similar, warm (cache hit) | 560 | 26.4 | 21.2x | 659 | 29.8 |
| index build from texts | 5,105 | 744 | 6.9x | 5,325 | 759 |
| top-10 query, engine only | 659 | 14.7 | 44.9x | 724 | 18.4 |

### All timings

| Case | Median ms | p95 ms | Min ms | Max ms | Timed runs | Warmup |
|---|---:|---:|---:|---:|---:|---:|
| find_similar_postings (python, cold) | 6,092 | 6,214 | 5,837 | 6,216 | 10 | 1 |
| find_similar_postings (python, warm) | 560 | 659 | 412 | 677 | 10 | 1 |
| find_similar_postings (native, cold) | 1,400 | 1,470 | 1,326 | 1,474 | 10 | 1 |
| find_similar_postings (native, warm) | 26.4 | 29.8 | 22.9 | 30.7 | 10 | 1 |
| index build from texts (python) | 5,105 | 5,325 | 5,049 | 5,326 | 10 | 1 |
| index build from texts (native) | 744 | 759 | 730 | 763 | 10 | 1 |
| top-10 query (python) | 659 | 724 | 423 | 730 | 10 | 1 |
| top-10 query (native) | 14.7 | 18.4 | 7.79 | 19.7 | 200 | 1 |
| ctypes call, no work (native) | 0.00 | 0.00 | 0.00 | 0.01 | 200 | 1 |

Case definitions:

- find_similar_postings (python, cold): similarity cache cleared before each call: DB fetch + index build + query + top-10 ORM fetch
- find_similar_postings (python, warm): cache hit: cache-key query + query + top-10 ORM fetch
- find_similar_postings (native, cold): similarity cache cleared before each call: DB fetch + index build + query + top-10 ORM fetch
- find_similar_postings (native, warm): cache hit: cache-key query + query + top-10 ORM fetch
- index build from texts (python): relevance.python_similarity_vectors: tokenize + df/idf + tf-idf dict per document
- index build from texts (native): native.Similarity: ICU tokenize + vocabulary + idf + CSR weights, from the same (id, text) pairs
- top-10 query (python): relevance.python_top_k over the prebuilt vectors: cosine of one target against every document, sorted
- top-10 query (native): Similarity.top_k(id, 10): one ctypes call, scan of every document, result list built in Python
- ctypes call, no work (native): Similarity.top_k(id, 0): lookup of the id and return; the wrapper's argument and result handling plus the ctypes crossing, no scan

### Memory

| Engine | Index RSS delta MB (median of samples) | Samples MB |
|---|---:|---|
| python | 881.9 | 881.9, 913.8, 876.0 |
| native | 151.9 | 151.9, 149.8, 153.5 |

Python / native memory ratio: 5.8x.
