# Stage 2 key-skills prompt

**Problem.** Stage 2 returns up to 8 free-text skill phrases per posting: 17% are compounds
("C/C++", "GD&T/Tolerance Analysis"), hundreds differ only by case or hyphens, and acronyms split from
their expansions, so downstream clustering has to repair what extraction could have normalized.

## Change

Only the key-skills paragraph of the prompt changed: one skill per entry (split `A/B` unless it is one
established name), standard short singular name with no qualifiers or versions, one spelling (the acronym
if that is the norm, never both), reuse the profile's spelling when the posting names that skill, order
by importance, cap raised to 12, broad skill areas included, soft skills and role titles excluded
(`candidate.py`).

## Method

`build_requests.py` builds the Stage 2 batch for a user's interest set with the candidate prompt (dry run
by default; `--submit` spends money). `compare.py` reads the finished batch and compares it with the
stored results; nothing is written to the database. `--control` re-runs the unchanged prompt and
`compare_control.py` compares the three, to separate the prompt's effect from run-to-run noise.

## Results (53 postings; reports/)

| | v1 | v2 |
|---|---|---|
| Compound phrases | 24 (9%) | 4 (1%) |
| Times each distinct phrase is reused | 1.39 | 1.94 |
| Phrases matching profile vocabulary exactly | 61 | 99 |
| Postings at their cap | 34 of 53 (cap 8) | 23 of 53 (cap 12) |

Triage verdicts changed on 9 of 34 postings (v1 to v2), but the *same* v1 prompt also disagrees with its
earlier results on 9 of 34, so the verdict movement is run-to-run noise, not the prompt. Identical Stage 2
runs differ on roughly a quarter of verdicts.

## Open item: pinning temperature

Repeatability would improve with `temperature=0`. `claude-sonnet-5` (Stage 3) rejects sampling parameters,
and the installed Anthropic SDK dropped the typed `temperature` argument, so for Stage 2 it would have to
be sent as a raw request field. `build_requests.py --temp0 --tag NAME` builds a batch that does this to
check whether the API accepts it; `compare_repeat.py` compares two such batches. Not applied to production.
