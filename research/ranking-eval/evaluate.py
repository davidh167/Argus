"""Step 3: leave-one-positive-out ranking evaluation + report.

Protocol (see README): every positive is its own query with that single
posting as the only relevant document; the candidate list is that positive
plus every UNLABELED posting (the other positives are removed so they can't
compete as false negatives). Because all scorers are profile-vs-posting,
each system's ranking is global; a query just asks "where did this one
posting land among the unlabeled pool?". Full ranking, no negative
sampling (sampled metrics can reorder systems -- Rendle, KDD 2020).

Two pools: `full` (everything) and `plausible` (Stage 1 != skip, i.e. what a
free title filter already lets through) -- the second is the harder, fairer
test, since ranking an Apple chip-design role below an ML role is easy.

Metrics from ranx (nDCG@k, MRR, Recall@k) plus AUC computed from rank.
95% CIs are percentile bootstrap over queries. Significance vs the best
real system is ranx's paired Student's t-test over queries.

Run:  PYTHONPATH=. poetry run python research/ranking-eval/evaluate.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
from ranx import Qrels, Run, compare, evaluate

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DATA, REPORTS = HERE / "data", HERE / "reports"
SEED = 11
N_BOOT = 2000
METRICS = ["ndcg@10", "ndcg@50", "mrr", "recall@10", "recall@50", "recall@100"]
SYSTEMS = ["random", "stage1", "tfidf", "bm25", "bm25_app", "embed", "rrf"]
CANARIES = ["canary_length", "canary_format"]
LABELS = {
    "random": "Random",
    "stage1": "Stage 1 title tier",
    "tfidf": "TF-IDF (app's relevance_score)",
    "bm25": "BM25 (bm25s)",
    "bm25_app": "BM25 (app implementation)",
    "embed": "Embedding cosine (nomic-embed-text)",
    "rrf": "RRF(BM25 + embedding)",
    "canary_length": "canary: document length",
    "canary_format": "canary: punctuation fraction",
}


EXTRA_LABELS = {
    "fts5_raw": "SQLite FTS5 BM25, skills-list query",
    "fts5_nostop": "SQLite FTS5 BM25, stopwords removed",
    "argus_dense": "Dense (in-process), skills-list query",
    "argus_hybrid100": "Hybrid RRF, top-100 lists (in-process)",
    "argus_rrf_full": "RRF over full rankings",
    "narr_bm25": "BM25, profile-narrative query",
    "narr_dense": "Dense, profile-narrative query",
    "narr_rrf_full": "RRF over full rankings, profile-narrative query",
}

LENGTH_TOL = 0.25  # length-matched pool: candidates within +-25% of the positive's length
MIN_MATCHED = 150  # widen the length window until at least this many candidates remain


def build(rows, scores, pool_mask, rng, length_matched=False):
    ids = np.array([r["id"] for r in rows])
    lengths = np.array([len(r["text"]) for r in rows])
    pos_idx = [i for i, r in enumerate(rows) if r["positive"] and pool_mask[i]]
    neg_idx = np.array([i for i, r in enumerate(rows) if not r["positive"] and pool_mask[i]])

    def candidates_for(p):
        if not length_matched:
            return neg_idx
        tol = LENGTH_TOL
        while True:
            near = neg_idx[np.abs(lengths[neg_idx] - lengths[p]) <= tol * lengths[p]]
            if len(near) >= MIN_MATCHED or tol > 10:
                return near
            tol *= 1.5

    cand_by_p = {p: candidates_for(p) for p in pos_idx}
    qrels = Qrels({f"q{ids[p]}": {str(ids[p]): 1} for p in pos_idx})
    runs, aucs = {}, {}
    for name, s in scores.items():
        run, auc = {}, []
        scale = float(s.std()) or 1.0
        for p in pos_idx:
            cand = np.append(cand_by_p[p], p)
            sc = s[cand] + rng.random(len(cand)) * 1e-9 * scale  # random tie-break, per query
            run[f"q{ids[p]}"] = {str(ids[c]): float(v) for c, v in zip(cand, sc)}
            rank = 1 + int((sc > sc[-1]).sum())
            auc.append((len(cand) - rank) / (len(cand) - 1))
        runs[name] = Run(run, name=name)
        aucs[name] = np.array(auc)
    return (
        qrels,
        runs,
        aucs,
        [int(ids[p]) for p in pos_idx],
        int(np.median([len(c) for c in cand_by_p.values()])),
    )


def ci(values, rng):
    boots = rng.choice(values, size=(N_BOOT, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def evaluate_pool(rows, scores, spec, rng):
    mask, length_matched = spec
    qrels, runs, aucs, pos_ids, n_neg = build(rows, scores, mask, rng, length_matched)
    table = {}
    for name, run in runs.items():
        per_q = evaluate(qrels, run, METRICS, return_mean=False)
        row = {m: ci(np.asarray(per_q[m]), rng) for m in METRICS}
        row["auc"] = ci(aucs[name], rng)
        table[name] = row
    real = [s for s in SYSTEMS if s != "random"]
    best = max(real, key=lambda s: table[s]["ndcg@50"][0])
    cmp_runs = [runs[s] for s in SYSTEMS]
    report = compare(
        qrels,
        cmp_runs,
        ["ndcg@50", "recall@50"],
        max_p=0.05,
        stat_test="student",
        make_comparable=True,
    )
    return {
        "table": table,
        "best": best,
        "n_queries": len(pos_ids),
        "n_unlabeled": n_neg,
        "compare": str(report),
        "per_query_rank_best": per_query_ranks(runs[best], pos_ids),
        "pos_ids": pos_ids,
    }


def per_query_ranks(run, pos_ids):
    out = {}
    for pid in pos_ids:
        docs = run.run[f"q{pid}"]
        order = sorted(docs, key=docs.get, reverse=True)
        out[pid] = (order.index(str(pid)) + 1, len(order))
    return out


def fmt(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def render(results, rows, meta):
    out = [
        "# Ranking evaluation: does the scorer rank the roles a user chose above the rest?\n",
        f"Snapshot: {meta['n_postings']} postings, {meta['n_positive']} positives (manual capture / star / active application), "
        f"profile version {meta['profile_version_id']}, exported {meta['exported_at'][:10]}.\n",
        "Mean over leave-one-out queries, 95% bootstrap CI in brackets. See README for protocol and caveats.\n",
    ]
    for pool, res in results.items():
        out.append(
            f"\n## Pool: `{pool}` — {res['n_queries']} queries vs a median of {res['n_unlabeled']} unlabeled candidates each\n"
        )
        out.append("| System | nDCG@10 | nDCG@50 | MRR | Recall@50 | Recall@100 | AUC |")
        out.append("|---|---|---|---|---|---|---|")
        for name in SYSTEMS + CANARIES:
            r = res["table"][name]
            out.append(
                f"| {LABELS[name]} | {fmt(r['ndcg@10'])} | {fmt(r['ndcg@50'])} | {fmt(r['mrr'])} | {fmt(r['recall@50'])} | {fmt(r['recall@100'])} | {fmt(r['auc'])} |"
            )
        out.append(f"\nBest real system by nDCG@50: **{LABELS[res['best']]}**.\n")
        out.append(
            "Paired significance (ranx, Student's t, p<0.05; superscripts mark systems significantly different):\n"
        )
        out.append("```\n" + res["compare"] + "\n```\n")
    return "\n".join(out) + "\n"


def main() -> None:
    rows = [json.loads(l) for l in open(DATA / "snapshot.jsonl")]
    meta = json.loads((DATA / "meta.json").read_text())
    z = np.load(DATA / "scores.npz")
    # Optional extra systems (one score per posting, same order as snapshot.jsonl), e.g. from
    # native/bench/compare_with_bm25s.py. Shown under their own key.
    extra = np.load(DATA / "extra_scores.npz") if (DATA / "extra_scores.npz").exists() else {}
    for k in extra.files if extra else []:
        SYSTEMS.append(k)
        LABELS[k] = EXTRA_LABELS.get(k, k)
    scores = {k: z[k] for k in SYSTEMS + CANARIES if k in z.files}
    scores.update({k: extra[k] for k in extra.files} if extra else {})
    rng = np.random.default_rng(SEED)
    allp = np.ones(len(rows), bool)
    plausible = np.array([r["stage1"] != "skip" for r in rows])
    pools = {
        "full": (allp, False),
        "plausible": (plausible, False),
        # Headline: controls for the length confound the canaries expose.
        "plausible_length_matched": (plausible, True),
    }
    results = {p: evaluate_pool(rows, scores, m, rng) for p, m in pools.items()}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "results.json").write_text(
        json.dumps(
            {
                p: {
                    "table": r["table"],
                    "best": r["best"],
                    "n_queries": r["n_queries"],
                    "n_unlabeled": r["n_unlabeled"],
                }
                for p, r in results.items()
            },
            indent=2,
        )
    )
    (REPORTS / "report.md").write_text(render(results, rows, meta))
    print((REPORTS / "report.md").read_text())


if __name__ == "__main__":
    main()
