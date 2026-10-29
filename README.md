# Argus

A local-first system for finding jobs worth applying to. Argus collects postings from company
career pages, filters them cheaply before spending anything on an LLM, scores fit against a
structured, versioned profile, and tracks what happens after you apply. You bring your own
Anthropic API key; your data stays on your machine.

## What it does

- **Collects postings** from Greenhouse, Lever and Ashby APIs, from a self-healing scraper for
  other careers pages, and from a Chrome extension that captures the page you are looking at.
- **Filters in stages, cheapest first.** Title phrases from your profile, then a small-model triage (Haiku),
  then a structured fit analysis (Sonnet). Both LLM stages run through the Batch API and show a
  cost estimate before anything is submitted.
- **Scores fit against a profile** (experience, projects, tiered skills, criteria). Profiles are
  versioned; every verdict records the version it was computed against, so a profile edit marks
  old verdicts stale instead of silently invalidating them.
- **Shows where you stand.** A readiness view buckets analyzed roles (apply now / solid match /
  stretch / build toward), and a skills view clusters what the market asks for, weighted toward the
  roles you chose, and judges each skill against your profile.
- **Tracks applications** (status, resume used, follow-ups) and exports to CSV.
- **Offers three surfaces** over one backend: a native desktop app (Tauri), an MCP server for
  conversational use, and the browser extension.

## Results

Measured on one user's data; each row links to the report or README it comes from.

| Question | Finding | Source |
|---|---|---|
| Ranking: BM25 vs the previous TF-IDF cosine | nDCG@50 0.216 [0.145, 0.294] vs 0.109 [0.067, 0.162]; BM25 also beat a local embedding model (0.103). 53 queries on a length-matched pool of a 3,024-posting corpus, with positive-unlabeled labels and a residual capture-style confound | [research/ranking-eval](research/ranking-eval/README.md) |
| Skill-phrase clustering | Pairwise F1 0.16 (previous greedy centroid @ 0.65) to 0.75 (average-link @ 0.80); precision 0.09 to 0.78, recall 0.77 to 0.71. 270 phrases, 56 gold SAME pairs, tuned and scored on the same phrases | [research/cluster-quality](research/cluster-quality/README.md) |
| Triage repeatability | Two identical Haiku 5.5 runs differ on 1 to 2 of 67 verdicts; Haiku 4.5 differs on about a quarter | [research/stage2-prompt](research/stage2-prompt/README.md) |
| Field-neutral triage prompt | Changed 14 of 67 verdicts on Haiku 5.5 against a noise floor of 1 to 2; skips fell from 20 to 10, almost all to `analyze` | [research/stage2-prompt](research/stage2-prompt/README.md) |
| Native core vs Ollama | Same weights; mean cosine 0.999999 over 300 skill phrases. Throughput 609 vs 160 texts/s on phrases, 76 vs 60 on 1,200-character excerpts; 7,897 phrases in 5.6 s vs about 49 s. FTS5 BM25 nDCG@50 0.208 vs 0.216 for the shipped ranker | [native/README.md](native/README.md), [native/reports/ollama_agreement.md](native/reports/ollama_agreement.md) |

Limits: one user's postings and choices, and small samples (tens of queries, hundreds of phrases), so
treat these as evidence for the design choices, not benchmarks.

## Design decisions

- **Filter in stages to control cost.** Free title rules, then Haiku triage, then Sonnet analysis; both
  paid stages use the Batch API, are capped (default 30), previewed with an estimate, and skip postings
  already in an unapplied batch (`app/services/batch_preview.py`, `stage2_batch.py`, `stage3_batch.py`).
- **Version the profile; mark verdicts stale.** Each verdict stores the profile version it used, so an
  edit flags old results instead of deleting them (`app/models/verdict.py` `is_stale`,
  `app/schemas/verdict.py` `build_verdict_out`).
- **BM25 over embeddings for ranking.** It beat the embedding baseline in the evaluation above, so the app
  uses `bm25s`; embeddings are kept for clustering and skill evidence (`app/services/relevance.py`).
- **Weight market demand by interest.** Manual captures, stars and active applications count; bulk
  crawled postings do not, because ~3,000 of them swamped 53 hand-picked ones when tried with a small
  base weight. The weights are judgment calls, not measured (`app/services/roadmap.py`, `interest_weights`).
- **Scraped text is data, not instructions.** Stage 2, Stage 3, roadmap and scraper prompts say so and
  wrap posting content as untrusted (`app/services/stage2.py`, `stage3.py`).
- **Local-first.** PostgreSQL in Docker, a gitignored personal profile, optional in-process embeddings
  so Ollama is not required (`docker-compose.yml`, `native/`).
- **Choose models by measurement, and keep the evidence.** Stage 2 moved to Haiku 5.5 after the
  repeatability comparison above; the reports are committed under `research/`.

## Architecture

```
Chrome extension ─┐
MCP server ───────┼─► FastAPI backend ─► PostgreSQL
Desktop app (Rust)┘        │
                           ├─► Anthropic API (Batch API for triage and analysis)
                           └─► embeddings: local Ollama, or in-process via the native core
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pipeline, profile versioning and ranking work.

## Quick start

Requirements: Docker, Python 3.12+, [Poetry](https://python-poetry.org/), and an Anthropic API key.
The desktop app additionally needs Rust and the Tauri CLI.

```bash
cp .env.example .env                       # then set ANTHROPIC_API_KEY
docker compose up -d                       # PostgreSQL
poetry install
poetry run alembic upgrade head
poetry run python scripts/generate_api_token.py   # paste the output into .env as API_TOKEN
cp profile_seed.yaml profile_seed.local.yaml      # fill in your own experience, projects, skills
poetry run python scripts/seed_profile.py profile_seed.local.yaml
poetry run uvicorn app.main:app --reload --reload-dir app   # API docs at http://localhost:8000/docs
```

`profile_seed.yaml` is a blank template; your real profile lives in `profile_seed.local.yaml`, which
is gitignored. Be honest in `skills.core` versus `skills.developing` versus `skills.coursework`: the
fit analysis weights them very differently.

Embeddings for the skills view default to a local [Ollama](https://ollama.com) server
(`ollama pull nomic-embed-text`). To run without Ollama, build the native core and set
`EMBEDDINGS_BACKEND=native` (see [native/README.md](native/README.md)).

### Desktop app

```bash
cd desktop && cargo tauri dev
```

`cargo tauri build` produces an unsigned app bundle; macOS will ask you to right-click and Open it the
first time.

### Browser extension

Chrome, Manifest V3: `chrome://extensions` → Developer mode → Load unpacked → select `extension/`,
then paste your `API_TOKEN` on the options page.

### MCP server

`mcp_server.py` serves the same operations over stdio. Copy `.mcp.json.example` to `.mcp.json` and fix
the paths to register it with an MCP client.

## Cost and safety

You pay for what you run. Triage with the small model costs fractions of a cent per posting and fit
analysis a few cents, both at Batch API pricing. Nothing runs automatically or unbounded: every
LLM-costing action is triggered by you, capped by default, and previewed with an estimate first. All
routes except `/health` require the bearer token.

## Security and privacy

- The API requires a bearer token (`API_TOKEN`) on every router; only `/health` is open
  (`app/auth.py`, `app/main.py`). The desktop app talks to `http://127.0.0.1:8000`
  (`desktop/src-tauri/src/lib.rs`), and the quick-start `uvicorn` command binds to loopback by default.
- `ANTHROPIC_API_KEY` and `API_TOKEN` live in `.env`, which is gitignored. A rejected key returns a
  clear 502 and is logged, rather than an opaque 500 (`app/main.py`).
- What leaves your machine: job text and profile text sent to Anthropic with your own key, plus the
  requests that crawl careers pages. Embeddings run locally (Ollama or the native core).
- Gitignored: `.env`, `profile_seed.local.yaml`, `*.pdf`, `.mcp.json`, and `research/*/data/` (derived from
  personal postings).
- Scraped postings are treated as untrusted data in the prompts, as noted above. This is a prompt-level
  mitigation, not a guarantee.

## Repository layout

| Path | Contents |
|---|---|
| `app/` | FastAPI backend: models, services, routers, crawlers, MCP server |
| `alembic/` | Database migrations |
| `desktop/` | Tauri desktop app (Rust commands over the backend API, plain JS frontend) |
| `extension/` | Chrome extension |
| `native/` | C++ retrieval core (SQLite FTS5, llama.cpp embeddings, usearch) with a C API and Rust bindings |
| `research/` | Evaluation harnesses and the experiments behind the ranking and clustering choices |
| `scripts/` | Setup and maintenance scripts |
| `tests/` | pytest suite |

## Tests

```bash
poetry run pytest
```

The suite runs against its own throwaway database (`<name>_test`, recreated each run), so it never
touches real data, and it makes no live API calls. The native core has its own C++ and Rust tests; see
`native/README.md`.

## Research

The evaluation harnesses and reports behind the Results table are in [research/README.md](research/README.md).

## Current limitations

- The prompts and title rules no longer assume a field, but they have only been exercised on one ML/AI
  profile. Stage 1 does nothing until the profile lists title phrases (`title_fast_track`, `title_skip`),
  and a role type the old prompt excluded by name now has to be in `skip_patterns`.
- The desktop app and the native core are developed and tested on macOS (Apple silicon) only.
- The native core is not wired into the desktop app yet; the backend can use it for embeddings.
- Evaluations in `research/` come from one user's postings and choices, so sample sizes are small.

## Notes

Built with the help of AI coding tools. MIT licensed.
