# Argus native core

An optional in-process retrieval library: keyword search, embeddings, vector search and a sparse TF-IDF
similarity index behind a stable C API, with Rust bindings. It exists so the app does not need a separate embedding server (Ollama) or, for
retrieval, a database server. It is **not** wired into the desktop app yet; today the Python backend
uses it for embeddings (`EMBEDDINGS_BACKEND=native`, the default), a persistent phrase-vector cache and the
"similar roles" ranking (`SIMILARITY_ENGINE=native`, the default).

The rule for this component is to use existing libraries and write only the glue, the interface and the
measurements:

| Need | Library |
|---|---|
| Keyword search (BM25) | SQLite FTS5 |
| Embedding inference | [llama.cpp](https://github.com/ggml-org/llama.cpp), GGUF `nomic-embed-text-v1.5` |
| Vector index | [usearch](https://github.com/unum-cloud/usearch) (HNSW, cosine) |
| Unicode (similarity tokenizer) | [ICU](https://icu.unicode.org/) (`icu4c`) |
| Tests / benchmarks / build | Catch2, Google Benchmark, CMake + Ninja |

## Layout

- `include/argus.h` — the C API: `argus_open[_with_vectors]`, `argus_upsert`, `argus_upsert_vector`,
  `argus_search_bm25 | dense | hybrid`, `argus_upsert_records | get_vector | get_text | meta_get | meta_set` (the index as a
  persistent key -> text/vector store), `argus_sim_*` (the similarity index below), `argus_embedder_open`,
  `argus_embed_batch`. Ownership is explicit
  (callers allocate result buffers), calls on a handle are serialized, nothing throws across the boundary.
- `src/` — index (`argus.cpp`), similarity (`sim.cpp`) and embedder (`embedder.cpp`).
- `rust/argus-core/` — safe Rust wrapper. `build.rs` builds the C++ with the `cmake` crate and generates
  bindings with `bindgen`.
- `tests/` (Catch2), `bench/` (Google Benchmark, plus scripts that feed the ranking harness),
  `tools/fetch_model.py` (downloads the model file and verifies its SHA-256).

## Design notes

- Documents live in an FTS5 table; vectors live in SQLite as blobs (the source of truth, transactional)
  and an in-memory usearch index is rebuilt from them on open. One file, no index-consistency problem.
- Free-text queries are quoted token by token, so FTS5 query-syntax characters can never raise errors.
- Hybrid search is reciprocal-rank fusion (k=60) of the top-100 keyword and top-100 dense candidates.
- The library is built with hidden symbol visibility and exports only the C API.
- llama.cpp's Metal backend asserts if the process exits with a model still loaded, so always close the
  embedder (the Rust wrapper's `Drop` does).

## Sparse TF-IDF similarity (`argus_sim_*`)

An in-memory index that reproduces `app/services/relevance.py` (the Python "similar roles" implementation)
exactly: tokens are the runs of Python `\w` characters in `text.lower()`, `tf = count / len(tokens)`,
`idf = log(n_docs / (1 + df))` (zero or negative for very common terms, which are kept), cosine over the
`tf * idf` vectors, 0.0 when either vector is empty. Usage is create, `argus_sim_add` every document,
`argus_sim_finalize`, then `argus_sim_topk(id, k)`; the index is rebuilt from rows by the caller whenever
its cache key changes (Postgres stays the system of record; nothing is persisted). It is a plain CSR scan
with a dense accumulator for the query document; no inverted index or SIMD, because the measurements in
`research/native-engine/reports/native_*.md` did not call for them.

- **Tie order.** Ties are broken by insertion order, which is what Python's stable `sorted()` does over a
  dict filled in the same order. Feed documents in the order the Python code iterates them.
- **Numerics.** `double` throughout, compiled with `-ffp-contract=off` (no fused multiply-add). Sums use
  Neumaier compensation, as Python 3.12's `sum()` does, so norms are bit-identical to Python's and scores
  agree to the last bit or so (the parity gates require 1e-9).
- **Unicode.** ICU does the lowercasing (`u_strToLower`, root locale: full case mapping including
  `U+0130` and the final-sigma rule, the whole string before splitting) and the character classes (Python
  `\w` is category L*, or any numeric value, or `_`). Pure-ASCII text takes a shortcut with the same result.
  Not ICU's `u_isalnum`/`u_isalpha`, which are different sets. `argus_text_tokenize` and
  `argus_unicode_probe` expose the tokenizer so tests can compare it with Python.
- **Unicode version skew.** Python 3.12 ships the Unicode 15.0 database; ICU ships its own (78.3: Unicode 17.0
  on the development machine). `tests/test_native_similarity.py` checks every code point from U+0000 to
  U+10FFFF (surrogates excluded) against `re.fullmatch(r"\w", chr(c))` and `chr(c).lower()`. Every code point
  that Python 3.12 assigns agrees exactly. The only disagreements are characters Unicode 15.0 does not
  define (category `Cn` in Python) that newer ICU already classifies: 9,661 `\w` and 55 lowercase
  differences with ICU 78.3. The test accepts disagreement only there, and fails if the library's Unicode
  is older than Python's. Text containing such characters (scripts and letters added after Unicode 15.0)
  can tokenize differently from Python 3.12 until Python's database catches up.

## Build and test

Requires CMake, Ninja, ICU, Catch2, Google Benchmark and llama.cpp (on macOS: `brew install cmake ninja
catch2 google-benchmark llama.cpp icu4c`; Homebrew's icu4c is keg-only, and CMake finds it under
`/opt/homebrew/opt/icu4c*` or at `-DICU_ROOT=...`; Linux: `libicu-dev`). The system `libicucore` on macOS
has no public headers and is not used. usearch is fetched by CMake.

`-DARGUS_WITH_LLAMA=OFF` builds the index and similarity code without llama.cpp (no embedder; the
`argus_embedder_*` symbols are absent and `app.native` says so if you ask for embeddings). CI builds that
configuration on Linux (`apt install cmake ninja-build libicu-dev libsqlite3-dev`) so the Python parity
tests run on every pull request; set `ARGUS_LIB_PATH` to use a library built elsewhere.

```bash
cmake -S native -B native/build -G Ninja -DARGUS_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build native/build && ctest --test-dir native/build          # 36 tests under ASan + UBSan (30 with -DARGUS_WITH_LLAMA=OFF)
cmake -S native -B native/build-release -G Ninja && cmake --build native/build-release
(cd native/rust/argus-core && cargo test)                            # 20 tests
python3 native/tools/fetch_model.py                                  # prints the model path
```

Embedder tests skip when no model is found (`ARGUS_MODEL`, or an Ollama `nomic-embed-text` blob).

The backend uses it by default; there is nothing to set if the release library is built
(`native/build-release/libargus.dylib`) and the model is fetched to its default directory by
`fetch_model.py`. `ARGUS_LIB_PATH`, `ARGUS_MODEL_PATH` and `ARGUS_INDEX_PATH` override the defaults, and
`EMBEDDINGS_BACKEND=ollama` opts back into a local Ollama server. A missing library or model raises with the
command to run; there is no fallback.

**Persistent vector cache** (`app/services/vector_cache.py`). Each phrase's vector is saved in an index file
keyed by a 64-bit BLAKE2b hash of the text, with the text stored beside it and compared on every lookup, so a
hash collision is logged and the colliding phrase is re-embedded instead of getting another phrase's vector.
The file is stamped with the model file's SHA-256 (re-hashed only when its size or mtime changes); opening it
with a different model raises and says to delete the index. Lookup order: in-process dict, index file, model.

## Results (macOS arm64, Apple clang 21, system SQLite with FTS5)

**Embeddings agree with Ollama's** (same weights) at cosine 0.999999 on average over 300 skill phrases
(worst case 0.999965) and to six decimals on 100 document excerpts. In-process throughput is 609 texts/s
versus 160 for Ollama on short phrases (most of the gap is Ollama's HTTP round trip) and 76 versus 60 on
1,200-character documents. Embedding all 7,897 distinct skill phrases takes 5.6 s versus about 49 s.

**Ranking quality** on the ranking harness (length-matched pool, 53 queries; see
[research/ranking-eval](../research/ranking-eval/README.md)):

| System | nDCG@50 |
|---|---|
| bm25s, skills-list query (used by the app) | 0.216 [0.145, 0.294] |
| SQLite FTS5 BM25, same query | 0.208 [0.147, 0.279] |
| Dense (in-process, usearch), skills-list query | 0.103 |
| Dense, natural-language profile query | 0.188 [0.117, 0.267] |
| RRF over full rankings, natural-language profile query | 0.232 [0.164, 0.307] |

FTS5 BM25 is statistically indistinguishable from the shipped `bm25s` ranker. The usearch dense results
reproduce the earlier Ollama + numpy brute-force scores exactly. A natural-language profile query
significantly improves dense search; at this sample size nothing significantly beats plain BM25.

**Speed.** FTS5 search takes 2.7 ms per query at 3,000 documents and 25 ms at 30,000 (synthetic corpus,
10-term query), slower than an in-memory index such as bm25s (~0.08 ms). The app runs one query per
rescore, so the case for this component is distribution and offline use, not retrieval speed.

**Similar roles** (synthetic corpora from `research/native-engine`, macOS arm64, 30 timed runs at 3,000
postings and 10 at 30,000; full tables and method in
[`research/native-engine/reports/native_synthetic_3000.md`](../research/native-engine/reports/native_synthetic_3000.md) and
`native_synthetic_30000.md`). The native and Python engines return identical top-10 lists in the parity gates
(scores bit-identical in those samples). Median time of `find_similar_postings`, Python then native: with a
warm cache 63.3 ms then 5.07 ms at 3,000 postings and 560 ms then 26.4 ms at 30,000; cold (database fetch, index
build, query) 590 ms then 139 ms and 6,092 ms then 1,400 ms. Index memory (resident-set growth) is about 90 MB
for the Python cache entry against 9.5 MB native at 3,000 postings, and 882 MB against 152 MB at 30,000. These
are synthetic corpora; the real-corpus parity and timing runs are the owner's.

## Limits

Not integrated into the desktop app; macOS arm64 only; the library and its llama.cpp dependencies still
need bundling and code-signing for distribution; no other embedding models compared; the model download
script has been exercised only against a local test server.
