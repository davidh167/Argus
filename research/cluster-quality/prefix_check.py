"""Does the nomic task prefix matter for phrase clustering? We currently
embed bare phrases (no prefix). The model card says to prefix every input.
Same gold/metric as run_methods.py; sweeps prefix x average-link threshold.
Also reports the random-pair similarity floor per prefix and a few
acronym/case probes.
Run: PYTHONPATH=. poetry run python research/cluster-quality/prefix_check.py"""

import json
from itertools import combinations
from pathlib import Path

import gold
import methods
import numpy as np

from app.services import embeddings

HERE = Path(__file__).parent
rows = json.loads((HERE / "data" / "phrases.json").read_text())
phrases = sorted({s for r in rows if r["interest"] >= 1.0 for s in r["skills"]})
labels = {
    (i, j): gold.label(phrases[i], phrases[j]) for i, j in combinations(range(len(phrases)), 2)
}


def f1(assign):
    tp = fp = fn = 0
    for (i, j), lab in labels.items():
        if lab is None:
            continue
        together = assign[i] == assign[j]
        tp += together and lab == "same"
        fp += together and lab == "diff"
        fn += (not together) and lab == "same"
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


rng = np.random.default_rng(0)
out = [
    "| prefix | random-pair p50 / p99 | "
    + " | ".join(f"F1@{t}" for t in (0.7, 0.75, 0.8, 0.85, 0.9))
    + " | best |",
    "|---|---|" + "---|" * 6,
]
for prefix in ("", "clustering: ", "classification: ", "search_query: ", "search_document: "):
    v = embeddings.embed_texts([prefix + p for p in phrases])
    idx = rng.integers(0, len(phrases), size=(3000, 2))
    sims = np.sum(v[idx[:, 0]] * v[idx[:, 1]], axis=1)[idx[:, 0] != idx[:, 1]]
    scores = []
    for t in (0.7, 0.75, 0.8, 0.85, 0.9):
        scores.append(f1(methods.agglomerative(phrases, v, t))[2])
    out.append(
        f"| `{prefix or '(none)'}` | {np.percentile(sims, 50):.2f} / {np.percentile(sims, 99):.2f} | "
        + " | ".join(f"{s:.3f}" for s in scores)
        + f" | {max(scores):.3f} |"
    )
print("\n".join(out))

print("\nacronym / case probes (cosine):")
probes = [
    ("NLP", "Natural language processing"),
    ("NLP", "nlp"),
    ("LLMs", "Large Language Models"),
    ("Go", "Golang"),
    ("RAG pipelines", "Retrieval Augmented Generation"),
    ("AI infrastructure", "AI policy"),
    ("C/C++", "Python"),
]
for prefix in ("", "clustering: "):
    v = embeddings.embed_texts([prefix + x for pair in probes for x in pair])
    print(" prefix", repr(prefix))
    for k, (a, b) in enumerate(probes):
        print(f"   {a!r} ~ {b!r}: {float(v[2 * k] @ v[2 * k + 1]):.3f}")
