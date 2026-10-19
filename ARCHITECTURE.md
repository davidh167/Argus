# Architecture

## Goals

Job hunting at volume is mostly reading postings that are obvious mismatches. Calling an LLM on every
one is expensive and noisy, so the design question is when *not* to call it. Argus filters in
stages, makes every paid call previewable and bounded, and records enough provenance that results stay
trustworthy as the profile changes.

## Components

- **Backend** (`app/`): FastAPI over PostgreSQL via SQLAlchemy, Alembic migrations. A service layer
  holds the logic; routers are thin. All routes require a bearer token generated locally.
- **Surfaces**: a Tauri desktop app (Rust commands that call the backend API; plain JS UI), an MCP
  server, and a Manifest V3 Chrome extension. None duplicates business logic.
- **Native core** (`native/`): optional in-process retrieval (C++20 with a C API and Rust bindings).

## Data model

| Table | Purpose |
|---|---|
| `companies`, `postings` | Postings keyed by company and external id. Company identity on shared ATS hosts is host plus first path segment, so two tenants of one host stay separate. |
| `verdicts` | One row per posting and profile version: Stage 1/2/3 results, the profile version each stage used, token counts. |
| `profile_versions` | Immutable snapshots of the profile. Editing creates a new version. |
| `applications` | One row per posting: status, resume used, dates, notes. |
| `batch_submissions` | Every Batch API submission, the postings it covers, and whether its results were applied. |
| `skill_roadmap`, `skill_judgments` | Skill gap briefs, and cached per-profile-version judgments of whether the profile demonstrates a skill. |
| `site_scrapers` | Per-domain extraction strategies for the scraper catalog. |
| `profile_intake_messages` | Chat history for building a profile from a conversation or a resume file. |

Structured columns are JSONB rather than Postgres arrays.

## The pipeline

1. **Ingest.** Greenhouse/Lever/Ashby APIs, the scraper catalog (a CSS-selector strategy generated once
   per domain by an LLM, reused deterministically, regenerated once if validation fails), and manual
   capture. Closed postings are detected when they disappear from a crawl.
2. **Stage 1: rules.** Title patterns classify a posting as fast-track, analyze or skip. Free.
3. **Stage 2: triage.** A small model reads the extracted JD sections against the profile criteria and
   returns fast-track, analyze or skip, plus up to 12 normalized key skills. Batch API.
4. **Stage 3: fit analysis.** A larger model returns a structured verdict (fit, seniority, hard and soft
   gaps, stack overlap, which base resume to use). Batch API.

Batches are capped (default 30, ranked by relevance to the profile), previewed with a cost estimate,
and tracked. A posting already in an unapplied batch is excluded from eligibility so it is never paid
for twice, and Stage 3 only takes postings that passed Stage 1 or 2 unless explicitly overridden.

## Profile versioning

Each verdict stores the profile version it was computed against. When the profile changes, old verdicts
are shown as stale and can be re-run on demand. Skill judgments are cached per profile version and
invalidated the same way.

## Ranking and skills

- **Relevance** of a posting to the profile is BM25 (`bm25s`) over title plus description. An
  evaluation (see `research/ranking-eval`) showed it beating a TF-IDF cosine baseline and the title
  tier on held-out manual picks.
- **Skills in demand** clusters the key skills extracted by Stage 2. Phrases from the roles the user
  chose or starred are clustered first (average-link at a threshold calibrated against labeled pairs,
  `research/cluster-quality`); other phrases join the nearest cluster. Clusters are weighted toward
  those chosen roles.
- **Have / developing / gap** for each skill is decided by retrieval followed by an LLM judgment:
  embeddings and BM25 shortlist the relevant profile evidence, a model decides, and the result is
  cached. Embedding similarity alone could not separate an umbrella skill the profile demonstrates
  from an unrelated gap.
- **Readiness tiers** bucket analyzed roles by hard-gap count.

## Testing

pytest against a dedicated, recreated database; Anthropic calls are always mocked (an autouse fixture
blanks the API key). The native core has Catch2 tests under AddressSanitizer/UBSan and Rust
integration tests.
