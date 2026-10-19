# Ranking evaluation

**Question.** Given a profile, which scorer ranks the postings a user deliberately picked above the rest
of a ~3,000-posting corpus: a TF-IDF cosine, a free title filter, BM25, a local embedding model, or a
fusion of them?

## Run

```bash
poetry install --with research
PYTHONPATH=. poetry run python research/ranking-eval/export_snapshot.py  # read-only DB -> data/
PYTHONPATH=. poetry run python research/ranking-eval/scorers.py          # needs Ollama + nomic-embed-text
PYTHONPATH=. poetry run python research/ranking-eval/evaluate.py         # -> reports/report.md, results.json
```

## Labels

A posting is a **positive** when it carries an explicit interest signal (manual capture, star, or an
active application). Everything else is **unlabeled, not negative**: a role a user never saw is not a role
they would reject. Dismissed postings would be real negatives, so the harness improves as dismissals
accumulate. Snapshot: 3,024 postings, 53 positives.

## Protocol

- **Leave-one-positive-out.** Each positive is its own query with that posting as the only relevant
  document. Candidates are that posting plus the unlabeled postings; the other positives are removed so
  they cannot compete as false negatives.
- **Full ranking, no negative sampling** (sampled metrics can reorder systems; Rendle, KDD 2020).
- **Metrics:** nDCG@10/50, MRR, Recall@50/100 via [ranx](https://pypi.org/project/ranx/), AUC from rank.
  95% intervals are a percentile bootstrap over queries; significance is ranx's paired Student's t-test.
- **Baselines:** the Stage 1 title tier, the previous TF-IDF `relevance_score`,
  [bm25s](https://pypi.org/p/bm25s), nomic-embed-text (with its documented task prefixes), RRF fusion.
- **No LLM verdicts as scorers:** Stage 3 exists only for postings the user chose to analyze, which is
  label leakage.

## Why three pools and two canaries

Every positive is a manual capture and every unlabeled posting was bulk-ingested, so a scorer can look
good by detecting how the text was captured rather than relevance. Two canaries score documents by length
and by punctuation fraction only. On the full corpus, length alone reached AUC 0.84, above the embedding
model, so early numbers were inflated. The headline pool is **length-matched**: candidates within ±25% of
the positive's length (widened to at least 150). On it the canaries fall to AUC 0.63 (length) and 0.65
(punctuation); only BM25 variants are clearly above that.

## Findings (length-matched pool, 53 queries; nDCG@50 with 95% interval)

| System | nDCG@50 | Recall@50 | AUC |
|---|---|---|---|
| Random | 0.036 | 0.113 | 0.52 |
| Stage 1 title tier | ~0.11-0.13 | 0.45 | 0.65 |
| TF-IDF cosine (previous scorer) | 0.109 [0.067, 0.162] | 0.377 | 0.63 |
| **BM25 (bm25s)** | **0.216 [0.145, 0.294]** | 0.509 | 0.81 |
| nomic-embed-text, skills-list query | 0.103 [0.053, 0.166] | 0.302 | 0.67 |
| RRF(BM25 + embedding) | 0.180 [0.110, 0.260] | 0.472 | 0.79 |
| BM25, natural-language profile query | 0.266 [0.179, 0.364] | 0.566 | 0.84 |
| Dense, natural-language profile query | 0.188 [0.117, 0.267] | 0.472 | 0.82 |
| RRF, natural-language profile query | 0.232 [0.164, 0.307] | 0.585 | 0.87 |

- BM25 is significantly better than the TF-IDF cosine, the title tier and the embedding baseline
  (paired test, p<0.05), so it replaced TF-IDF for ranking.
- A bare skills list is a poor *dense* query. A natural-language description of the profile significantly
  improves dense search (0.103 to 0.188). For BM25 the gain is not significant at this sample size.
- Nothing significantly beats plain BM25 at 53 queries.
- The Stage 1 baseline has heavy ties (three tiers), so its number moves ~0.01 between runs.

## Limits

Positive-unlabeled labels, 53 queries, one user's profile, and a residual capture-style confound that the
length-matched pool only partly removes.

Rows for SQLite FTS5 and the in-process dense/hybrid systems come from [`native/`](../../native/README.md)
via `data/extra_scores.npz`.
