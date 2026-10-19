"""Score clustering methods against the author-labeled gold on the
interest-set phrases (data/phrases.json from export_phrases.py).
Run: PYTHONPATH=. poetry run python research/cluster-quality/run_methods.py
Writes reports/methods.md."""

import json
from itertools import combinations
from pathlib import Path

import gold
import methods
import numpy as np

from app.services import embeddings

HERE = Path(__file__).parent
INTEREST_MIN = 1.0

rows = json.loads((HERE / "data" / "phrases.json").read_text())
phrases = sorted({s for r in rows if r["interest"] >= INTEREST_MIN for s in r["skills"]})
vecs = embeddings.embed_texts(phrases)

# Gold members that are not in the data would silently weaken the gold.
in_gold = {p for g in gold.SAME for p in g}
missing = sorted(in_gold - set(phrases))
assert not missing, f"gold phrases not in data: {missing}"

pairs = [(i, j) for i, j in combinations(range(len(phrases)), 2)]
labels = {(i, j): gold.label(phrases[i], phrases[j]) for i, j in pairs}
n_same = sum(v == "same" for v in labels.values())
n_ignored = sum(v is None for v in labels.values())


def score(assign):
    tp = fp = fn = 0
    fps, fns = [], []
    for (i, j), lab in labels.items():
        if lab is None:
            continue
        together = assign[i] == assign[j]
        if together and lab == "same":
            tp += 1
        elif together:
            fp += 1
            fps.append((phrases[i], phrases[j]))
        elif lab == "same":
            fn += 1
            fns.append((phrases[i], phrases[j]))
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1, fp, fn, fps, fns


configs = []
for t in (0.65, 0.70, 0.75, 0.80, 0.85):
    configs.append((f"greedy centroid @{t}", methods.greedy_centroid, t))
    configs.append(
        (
            f"greedy centroid + normalize @{t}",
            methods.with_normalization(methods.greedy_centroid),
            t,
        )
    )
for t in (0.65, 0.70, 0.75, 0.80, 0.85):
    configs.append((f"agglomerative avg-link @{t}", methods.agglomerative, t))
    configs.append(
        (
            f"agglomerative avg-link + normalize @{t}",
            methods.with_normalization(methods.agglomerative),
            t,
        )
    )

for t in (0.75, 0.80, 0.85):
    configs.append(
        (
            f"agglomerative + normalize + acronyms @{t}",
            methods.with_acronyms(methods.with_normalization(methods.agglomerative)),
            t,
        )
    )

import canonicalize

canon = canonicalize.canon_all(phrases, "llama3.2:3b")
canon_vecs = embeddings.embed_texts([canon[p] for p in phrases])
# similarity = max of raw-phrase and canonical-name similarity
both = np.concatenate([vecs, canon_vecs], axis=1) / np.sqrt(
    2
)  # cosine of concat == mean of the two cosines


def on_canon(method):
    return lambda ph, v, t: method(ph, canon_vecs, t)


def on_both(method):
    return lambda ph, v, t: method(ph, both, t)


for t in (0.75, 0.80, 0.85, 0.90):
    configs.append(
        (f"agglomerative on LLM-canonical names @{t}", on_canon(methods.agglomerative), t)
    )
    configs.append(
        (f"agglomerative on raw+canonical concat @{t}", on_both(methods.agglomerative), t)
    )

out = [
    "# Cluster quality vs author-labeled gold",
    "",
    f"{len(phrases)} distinct interest-set phrases; {n_same} gold SAME pairs; {n_ignored} NEAR pairs excluded.",
    "Gold was labeled with LLM assistance and reviewed by the author -- treat small differences as noise.",
    "",
    "| method | clusters | precision | recall | F1 | FP | FN |",
    "|---|---|---|---|---|---|---|",
]
detail = {}
for name, fn, t in configs:
    assign = fn(phrases, vecs, t)
    p, r, f1, fp, fn_, fps, fns = score(assign)
    out.append(f"| {name} | {len(set(assign))} | {p:.3f} | {r:.3f} | {f1:.3f} | {fp} | {fn_} |")
    detail[name] = (fps, fns)

best = max(detail, key=lambda k: 0)  # placeholder; pick by F1 below
rows_f1 = {}
for name, fn, t in configs:
    rows_f1[name] = score(fn(phrases, vecs, t))[2]
best = max(rows_f1, key=rows_f1.get)
out += [
    "",
    f"Best by F1: **{best}** ({rows_f1[best]:.3f})",
    "",
    "False positives (merged but not SAME), best method:",
]
out += [f"- {a} ~ {b}" for a, b in detail[best][0][:25]]
out += ["", "False negatives (SAME but split), best method:"]
out += [f"- {a} / {b}" for a, b in detail[best][1][:25]]
(HERE / "reports").mkdir(exist_ok=True)
(HERE / "reports" / "methods.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
