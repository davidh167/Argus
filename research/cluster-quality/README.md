# Skill clustering quality

**Question.** Stage 2 extracts short skill phrases ("distributed systems design", "Postgres"). Which
clustering of those phrases best matches how a person would group them?

## Method

- Phrases: the 270 distinct key-skill phrases from a user's 53 chosen postings.
- Gold (`gold.py`): pairs labeled SAME (strict equivalence, 56 pairs), NEAR (related or hierarchical,
  518 pairs, excluded from scoring) or DIFFERENT (everything else). Labels were written with LLM
  assistance and reviewed by the author; treat small differences as noise. The gold was amended once
  after inspecting the first false positives (hierarchy pairs that had not been listed).
- Embeddings: nomic-embed-text. Metric: pairwise precision/recall/F1 (`run_methods.py` writes
  `reports/methods.md`).

## Findings

| Method | Clusters | Precision | Recall | F1 |
|---|---|---|---|---|
| Greedy centroid @ 0.65 (previous setting) | 110 | 0.09 | 0.77 | 0.16 |
| Greedy centroid @ 0.85 | 222 | 0.86 | 0.54 | 0.66 |
| **Average-link @ 0.80** | 203 | 0.78 | 0.71 | **0.75** |
| Average-link @ 0.80 + acronym rules | 199 | 0.70 | 0.80 | 0.75 |

1. The previous threshold was calibrated on random and hand-picked pairs, which never included
   same-head-word pairs ("AI infrastructure" / "AI policy" score 0.77). On the user's phrases the
   random-pair 99th percentile is 0.71, above the old 0.65 cutoff. Calibrate on the population being
   clustered.
2. Order-independent average-link at ~0.80 is the best embedding-only setting.
3. Embeddings cannot see aliases (Go/Golang 0.62, LLMs/Large Language Models 0.44). A deterministic
   initialism rule raises recall but gives back precision; asking a 3B local model to canonicalize
   phrases made things worse (it expanded "RAG" to "rapid application development").
4. The nomic task prefix shifts the whole similarity scale (random-pair median 0.39 with no prefix, 0.67
   with `clustering:`) for a small F1 change that is within noise (`prefix_check.py`). A threshold is only
   meaningful together with its model and prefix.
5. Remaining "false positives" are mostly hierarchy (LLM serving / LLMs).

## Limits

270 phrases, 56 gold pairs, one labeler plus review, and the method and threshold were tuned and scored
on the same phrases (the later Stage 2 prompt change produces fresh phrases for re-checking).
