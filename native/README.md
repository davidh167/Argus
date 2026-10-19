# Argus native core

An optional in-process retrieval library: keyword search, embeddings and vector search behind a stable C
API, with Rust bindings. It exists so the app does not need a separate embedding server (Ollama) or, for
retrieval, a database server. It is **not** wired into the desktop app yet; today the Python backend can
use it for embeddings (`EMBEDDINGS_BACKEND=native`).

The rule for this component is to use existing libraries and write only the glue, the interface and the
measurements:

| Need | Library |
|---|---|
| Keyword search (BM25) | SQLite FTS5 |
| Embedding inference | [llama.cpp](https://github.com/ggml-org/llama.cpp), GGUF `nomic-embed-text-v1.5` |
| Vector index | [usearch](https://github.com/unum-cloud/usearch) (HNSW, cosine) |
| Tests / benchmarks / build | Catch2, Google Benchmark, CMake + Ninja |

## Layout

- `include/argus.h` — the C API: `argus_open[_with_vectors]`, `argus_upsert`, `argus_upsert_vector`,
  `argus_search_bm25 | dense | hybrid`, `argus_embedder_open`, `argus_embed_batch`. Ownership is explicit
  (callers allocate result buffers), calls on a handle are serialized, nothing throws across the boundary.
- `src/` — index (`argus.cpp`) and embedder (`embedder.cpp`).
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

## Build and test

Requires CMake, Ninja, Catch2, Google Benchmark and llama.cpp (on macOS: `brew install cmake ninja catch2
google-benchmark llama.cpp`). usearch is fetched by CMake.

```bash
cmake -S native -B native/build -G Ninja -DARGUS_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build native/build && ctest --test-dir native/build          # 21 tests under ASan + UBSan
cmake -S native -B native/build-release -G Ninja && cmake --build native/build-release
(cd native/rust/argus-core && cargo test)                            # 20 tests
python3 native/tools/fetch_model.py                                  # prints the model path
```

Embedder tests skip when no model is found (`ARGUS_MODEL`, or an Ollama `nomic-embed-text` blob).

To use it from the backend, set in `.env`: `EMBEDDINGS_BACKEND=native` and `ARGUS_MODEL_PATH=<model.gguf>`
(`ARGUS_LIB_PATH` defaults to `native/build-release/libargus.dylib`).

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

## Limits

Not integrated into the desktop app; macOS arm64 only; the library and its llama.cpp dependencies still
need bundling and code-signing for distribution; no other embedding models compared; the model download
script has been exercised only against a local test server.
