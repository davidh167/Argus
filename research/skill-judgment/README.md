# Skill judgment: have, developing or gap

**Problem.** The skills view must say whether a profile demonstrates each skill the market asks for.
Embedding similarity between a skill and the profile's entries cannot: exact tool names match at 1.0, but
umbrella skills the profile clearly demonstrates ("LLMs", "NLP") score below false matches ("C/C++" 0.76).
Against 22 labeled clusters no similarity cutoff from 0.65 to 0.85 got more than 15 right.

**Approach** (`app/services/skill_judgment.py`). Retrieve a shortlist of profile evidence for the skill
(top 8 by embedding plus top 4 by BM25 over listed skills, projects, experience and open-source work),
then ask a model to decide have / developing / gap and cite the evidence it used. Citations outside the
shortlist are dropped. Judgments are cached per profile version.

**Run** (dry run is free; `--run` makes one synchronous call per labeled cluster, a few cents):

```bash
PYTHONPATH=. poetry run python research/skill-judgment/run.py [--run]
```

The labels describe one person's profile, so they live in `data/labels.json` (gitignored); see
`labels.example.json` for the format. `reports/judgment.md` records only the aggregate.

**Result.** 22/22 against the author's labels, versus 14-15/22 for the best similarity cutoff.

**Limits.** 22 clusters, one profile, and the prompt and the labels were written by the same person with
LLM assistance. It shows the approach is viable, not that it is calibrated.
