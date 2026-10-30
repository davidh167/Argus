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
be sent as a raw request field (`build_requests.py --temp0 --tag NAME`, compared with `compare_repeat.py`).
Claude Haiku 5.5 only accepts `temperature=1`, so on that model pinning is not possible and the question
becomes whether the model is more consistent on its own.

## Comparing models (Haiku 5.5)

`build_requests.py --model claude-haiku-5-5 --tag h55a --submit` (and `--tag h55b`) re-runs the unchanged
production prompt and tool on another model. `compare_models.py h55a h55b` then compares those runs with
the stored production verdicts and with `batch_control.json`, using the production model's own
run-to-run disagreement (stored vs control, above) as the yardstick. It also reports errored requests,
refusals, outputs truncated at `max_tokens` (Haiku 5.5's tokenizer produces about 30% more tokens) and
token use. At the Batch API rate a run costs under a cent of input tokens on Haiku 5.5.

### Result: Haiku 5.5, and the field-neutral prompt (67 interest-set postings)

- Haiku 5.5 is far more repeatable than Haiku 4.5: two identical runs differ on 1 to 2 of 67 verdicts,
  against about a quarter for 4.5. At the production limit of `max_tokens=256` it truncates 19 to 29 of 67
  outputs (every one missing a required field); at 1024 none are truncated. It disagrees with 4.5 on
  roughly half the postings, mostly by skipping fewer and rarely choosing `fast_track`.
- Removing the ML/AI wording from the Stage 2 prompt (deferring to the profile's `target_signals` and
  `skip_patterns`) changed 14 of 67 verdicts on Haiku 5.5, against a noise floor of 1 to 2: skips fell from
  20 to 10, almost all of them to `analyze`. Anything the old prompt excluded by name (cybersecurity,
  sales-adjacent or forward-deployed roles) now has to be in the user's `skip_patterns`.


## Canary: neutralised key_skills wording (tag `neutral2`)

`app/services/stage2.py` no longer cites LLM/NLP/CUDA/RAG/"machine learning" in the key_skills guidance
(examples now span clinical, trades, finance and software). Before trusting it, run it once and compare
with the stored production verdicts and the earlier unchanged-prompt run (`batch_control.json`, whose
disagreement with the stored verdicts is the run-to-run noise yardstick). Nothing below is run for you;
only `--submit` spends money.

```
# 1. Build and inspect (dry run: request count and token estimate, nothing sent, nothing written)
PYTHONPATH=. poetry run python research/stage2-prompt/build_requests.py --control --tag neutral2

# 2. Submit (costs money; saves the batch id to data/batch_neutral2.json)
PYTHONPATH=. poetry run python research/stage2-prompt/build_requests.py --control --tag neutral2 --submit

# 3. After the batch has ended: stored vs control vs neutral2, with errors/refusals/truncation and tokens
PYTHONPATH=. poetry run python research/stage2-prompt/compare_models.py neutral2
```

`--control` is required: it means "use the production prompt in `app/services/stage2.py`", which is now the
neutral one. Without it the script builds the old v2 candidate prompt. Read the table as: `stored vs control`
is the noise floor; `control vs neutral2` and `stored vs neutral2` should be about that size. Verdict
differences well above it mean the wording matters. The key_skills themselves are not compared by
`compare_models.py`; use `compare.py`-style inspection of the batch results for spelling consistency.
