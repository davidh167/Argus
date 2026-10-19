# Argus

A local-first system for finding jobs worth applying to. Argus collects postings from company
career pages, filters them cheaply before spending anything on an LLM, scores fit against a
structured, versioned profile, and tracks what happens after you apply. You bring your own
Anthropic API key; your data stays on your machine.

## What it does

- **Collects postings** from Greenhouse, Lever and Ashby APIs, from a self-healing scraper for
  other careers pages, and from a Chrome extension that captures the page you are looking at.
- **Filters in stages, cheapest first.** Regex title rules, then a small-model triage (Haiku),
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

`research/` holds the measurements behind several design choices, for example that BM25 ranks the
roles a user picked above a TF-IDF baseline, and how skill phrases should be clustered. See
[research/README.md](research/README.md).

## Current limitations

- The Stage 1-3 prompts and title rules are tuned for an ML/AI job search; making them field-agnostic is
  planned but not done.
- The desktop app and the native core are developed and tested on macOS (Apple silicon) only.
- The native core is not wired into the desktop app yet; the backend can use it for embeddings.
- Evaluations in `research/` come from one user's postings and choices, so sample sizes are small.

## Notes

Built with the help of AI coding tools. MIT licensed.
