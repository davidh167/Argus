# Native engine: Python baseline (Phase 0) and native similarity (Phase 2)

Phase 2 (native sparse TF-IDF similarity) is at the bottom of this file; everything above it is Phase 0.

**Question.** Where does the time go in the current Python similar-roles, relevance-ranking and
market-demand code, and is a native rewrite of the similar-roles path worth its speed claim?

This harness measures the Python implementation only. It contains no native code and calls no model.

## What is timed

| Case | Entry point |
|---|---|
| `find_similar_postings (cold)` | `relevance.find_similar_postings`, similarity cache cleared before every call |
| `find_similar_postings (warm)` | same call with the cache populated |
| `rescore_all` | `relevance.rescore_all` (rolled back after each run) |
| `bm25_scores` | `relevance.bm25_scores` over every posting, profile query as the query |
| `market_demand (stub embedder)` | `roadmap.market_demand` with `embeddings.embed_texts` replaced by a deterministic stub |
| `component: ...` | the cold similar-roles cost split into DB fetch, tokenize, df/idf + vectors, cosine scan (re-composed from the same `relevance` functions) |

Each case runs `--warmup` untimed runs, then `--runs` timed runs (default 30, minimum 10). `find_similar`
cycles through a fixed seeded sample of 50 posting ids so one posting does not set the number. The
report gives median and p95 (numpy linear interpolation), min, max, N, machine, Python and library
versions, Postgres version and the git commit.

The stub embedder (`bench_stub.py`) maps each word to a fixed pseudo-random 768-d vector and a phrase to
the normalized sum, so equal text gives equal vectors in every process and phrases that share words are
close. It times the clustering code, not embedding, and its similarity distribution is not the real
model's, so the `market_demand clusters` count printed beside the table describes the stub, not real data.

## Corpora

- **Synthetic** (`--synthetic N`): seeded (`bench_corpus.DEFAULT_SEED`), reproducible by anyone. Zipf
  word frequencies over topic vocabularies, a long tail of rare terms, ~400-word postings, 10% near-duplicate
  postings within a company, ~6% of postings with non-ASCII words (accents, CJK, Cyrillic), 2% positives,
  and 4-7 key-skill phrases per posting at roughly the real distinct-to-total ratio. It is plausible in
  shape, not in meaning; real text will have different token and similarity distributions.
- **Snapshot** (`--snapshot DIR`): the `snapshot.jsonl` + `meta.json` written by
  `research/ranking-eval/export_snapshot.py` (gitignored `research/ranking-eval/data/`). Real titles and JD
  text; the profile's `target_signals` is the real profile query. The snapshot has no `stage2_key_skills`,
  so the market_demand input is synthesized (same generator as above) and the report says so.

Both are loaded into a throwaway Postgres database the harness creates and drops (`bench_db.py`):
`<configured db>_bench` by default, or `--db-name`. A name equal to the configured database is refused,
the URL is switched before `app.db` builds its engine, and the script stops if `app.db` was already
imported. `raw_jd_text` is loaded equal to the normalized text (real raw text is longer, so ORM loads
in `rescore_all` and `market_demand` are a lower bound).

## Non-ASCII scan (decision D4)

For every posting, the scan computes the top-10 similar postings twice with the same tf-idf formulas as
`relevance.py`: tokens from Python's Unicode `\w+` (current behavior) and from ASCII-only `\w+`
(`re.ASCII`, what a simple C++ tokenizer would do). It reports how many postings and tokens contain
non-ASCII word characters and how many postings' top-10 list (order, set, top-1) differs. It uses scipy
sparse matrices, not the Python dict path; `tests/test_native_engine_bench.py` checks it returns the
same neighbors as `find_similar_postings`. Similarities are rounded to 12 decimals and ties broken by
corpus order in both runs. A posting's list can change because a neighbor's text contains non-ASCII
tokens, not only its own, so the report also shows how many changed postings contain non-ASCII text.

## Run

From the repo root, with Postgres up (`docker compose up -d`):

```bash
# Synthetic, reproducible by anyone
PYTHONPATH=. poetry run python research/native-engine/run_baseline.py \
    --synthetic 3000 --out research/native-engine/reports/baseline_synthetic_3000.md
PYTHONPATH=. poetry run python research/native-engine/run_baseline.py \
    --synthetic 30000 --runs 10 --warmup 1 \
    --out research/native-engine/reports/baseline_synthetic_30000.md

# Real corpus (after export_snapshot.py wrote research/ranking-eval/data/)
PYTHONPATH=. poetry run python research/native-engine/run_baseline.py \
    --snapshot research/ranking-eval/data \
    --out research/native-engine/reports/baseline_snapshot.md
```

At 30,000 postings one `market_demand` run takes about a minute and `rescore_all` about 20 seconds, so
that size uses 10 timed runs (`--runs` accepts 10 or more; the report states when it is below 30). The
report is written only when all corpora finish, and per-case medians are printed to stderr as they
complete. Close other heavy programs while it runs; timings are wall clock on a shared machine.

## Reports

- [`reports/baseline_synthetic_3000.md`](reports/baseline_synthetic_3000.md) and
  [`reports/baseline_synthetic_30000.md`](reports/baseline_synthetic_30000.md): synthetic corpora.
- `reports/baseline_snapshot.md`: written by the owner's run on the real snapshot.

Each report states the Phase 0 kill criterion mechanically from its own numbers: warm `find_similar`
p95 < 50 ms and cold `find_similar` p95 < 1 s.

## Phase 2: native similarity engine

`app/native` and `native/src/sim.cpp` implement the "similar roles" ranking natively
(`SIMILARITY_ENGINE=native`, the default). The Python implementation stays in `app/services/relevance.py`
(`SIMILARITY_ENGINE=python`) as the reference; it is a parity/test switch, not a fallback.

| Script | Does |
|---|---|
| `bench_parity.py` | the comparison itself: Python top-k vs native top-k for the same documents in the same order (ids, order, scores within 1e-9); used by the tests and the two scripts below |
| `parity_snapshot.py` | parity for EVERY posting of a real exported snapshot (or `--synthetic N`), markdown report with counts and the first mismatches |
| `run_native.py` | parity gate on a seeded sample, then timings: `find_similar_postings` cold and warm under each engine, index build from texts, one top-10 query, ctypes call overhead, and index memory in fresh subprocesses (`bench_memory.py`) |

Both need the built library (`native/README.md`) and the throwaway Postgres database, like Phase 0. From the
repo root:

```bash
# Synthetic, reproducible by anyone
PYTHONPATH=. poetry run python research/native-engine/run_native.py \
    --synthetic 3000 --out research/native-engine/reports/native_synthetic_3000.md
PYTHONPATH=. poetry run python research/native-engine/run_native.py \
    --synthetic 30000 --runs 10 --warmup 1 --out research/native-engine/reports/native_synthetic_30000.md

# Real corpus (after export_snapshot.py wrote research/ranking-eval/data/)
PYTHONPATH=. poetry run python research/native-engine/parity_snapshot.py \
    --snapshot research/ranking-eval/data --out research/native-engine/reports/parity_snapshot.md
PYTHONPATH=. poetry run python research/native-engine/run_native.py \
    --snapshot research/ranking-eval/data --out research/native-engine/reports/native_snapshot.md
```

`run_native.py` stops before timing if its parity sample has a single mismatch. The exhaustive code-point,
tricky-string and whole-corpus parity tests are `tests/test_native_similarity.py`.

Reports: [`reports/native_synthetic_3000.md`](reports/native_synthetic_3000.md) and
[`reports/native_synthetic_30000.md`](reports/native_synthetic_30000.md) (written by the commands above);
`parity_snapshot.md` and `native_snapshot.md` are written by the owner's run on the real snapshot.
