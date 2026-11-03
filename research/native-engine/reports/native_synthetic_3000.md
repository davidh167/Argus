# Native sparse TF-IDF similarity vs the Python implementation (Phase 2)

- generated: 2026-11-03 09:48 UTC
- machine: macOS-27.0.1-arm64-arm-64bit, arm64, 10 logical CPUs
- python: 3.12.2
- libraries: numpy 2.5.3, scipy 1.18.1, bm25s 0.3.14, sqlalchemy 2.0.52
- postgres: PostgreSQL 17.11
- git commit: e20f39f
- native library: /Users/dhern162/argus/.claude/worktrees/agent-a59589817fdbb857a/native/build-release/libargus.dylib
- ICU: 78.3 (Unicode 17.0)

Medians and p95 are over the timed runs (linear interpolation, numpy default), warmup runs excluded. Times are wall clock around the call; database calls go to a local throwaway Postgres. Native and Python rank the same documents in the same order; equality of their results is checked by the parity gate above each table and by tests/test_native_similarity.py (every top-10 list identical, scores within 1e-9), not assumed. The 'warm' rows are the production comparison: Python's cache-hit path against the native cache-hit path. 'Median ratio' is the Python median divided by the native median.

Memory is the resident-set growth of a fresh process (corpus texts already in memory, library and ICU data already touched) between before and after building the index and a full garbage collection, measured with `ps -o rss`. The Python figure is the dict-of-dicts the cache holds; the native figure is the CSR index. RSS is page-granular and includes allocator slack, so small differences are noise.

## Corpus: synthetic-3000 (N = 3,000)

- seed: 20261101
- non-ASCII posting fraction: 0.06
- parity gate (run before timing): 100 query postings (seeded sample) x top-10 against the Python implementation fed in the same order: 0 id/order mismatches, 0 score mismatches (tolerance 1e-09), max |score diff| 0, 1000 of 1000 scores bit-identical
- fewest timed runs in any case: 30

### Native vs Python

| What | Python median ms | Native median ms | Median ratio | Python p95 ms | Native p95 ms |
|---|---:|---:|---:|---:|---:|
| find_similar, cold (DB fetch + build + query) | 590 | 139 | 4.3x | 639 | 164 |
| find_similar, warm (cache hit) | 63.3 | 5.07 | 12.5x | 78.7 | 6.46 |
| index build from texts | 484 | 73.8 | 6.6x | 504 | 74.5 |
| top-10 query, engine only | 56.3 | 1.50 | 37.4x | 71.4 | 1.97 |

### All timings

| Case | Median ms | p95 ms | Min ms | Max ms | Timed runs | Warmup |
|---|---:|---:|---:|---:|---:|---:|
| find_similar_postings (python, cold) | 590 | 639 | 563 | 661 | 30 | 3 |
| find_similar_postings (python, warm) | 63.3 | 78.7 | 50.1 | 84.5 | 30 | 3 |
| find_similar_postings (native, cold) | 139 | 164 | 130 | 182 | 30 | 3 |
| find_similar_postings (native, warm) | 5.07 | 6.46 | 4.21 | 7.02 | 30 | 3 |
| index build from texts (python) | 484 | 504 | 471 | 554 | 30 | 3 |
| index build from texts (native) | 73.8 | 74.5 | 72.6 | 74.9 | 30 | 3 |
| top-10 query (python) | 56.3 | 71.4 | 45.7 | 77.3 | 30 | 3 |
| top-10 query (native) | 1.50 | 1.97 | 1.08 | 2.18 | 200 | 3 |
| ctypes call, no work (native) | 0.00 | 0.00 | 0.00 | 0.01 | 200 | 3 |

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
| python | 89.5 | 89.5, 89.5, 87.4 |
| native | 9.5 | 11.6, 8.4, 9.5 |

Python / native memory ratio: 9.4x.
