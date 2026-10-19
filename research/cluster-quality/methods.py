"""Clustering methods under test. Each takes (phrases, vectors) and returns
a list[int] cluster label per phrase."""

import re

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


def greedy_centroid(phrases, vecs, threshold):
    """The app's current algorithm (_cluster_items_by_embedding), in input order."""
    centroids, sums, counts, labels = [], [], [], []
    for v in vecs:
        if centroids:
            sims = np.array(centroids) @ v
            j = int(np.argmax(sims))
            if sims[j] >= threshold:
                sums[j] += v
                counts[j] += 1
                c = sums[j] / counts[j]
                centroids[j] = c / np.linalg.norm(c)
                labels.append(j)
                continue
        centroids.append(v.copy())
        sums.append(v.copy())
        counts.append(1)
        labels.append(len(centroids) - 1)
    return labels


def agglomerative(phrases, vecs, threshold, method="average"):
    """Order-independent hierarchical clustering, cut at cosine sim = threshold."""
    if len(phrases) == 1:
        return [0]
    z = linkage(vecs, method=method, metric="cosine")
    return list(fcluster(z, t=1 - threshold, criterion="distance"))


def normalize(p: str) -> str:
    p = p.casefold().replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", p).strip()


def with_normalization(method, **kw):
    """Collapse phrases that are identical after casefold/hyphen
    normalization BEFORE clustering, then expand labels back."""

    def run(phrases, vecs, threshold):
        keys = [normalize(p) for p in phrases]
        first = {}
        for i, k in enumerate(keys):
            first.setdefault(k, i)
        idx = list(first.values())
        sub = method([phrases[i] for i in idx], vecs[idx], threshold, **kw)
        lab = {keys[i]: sub[n] for n, i in enumerate(idx)}
        return [lab[k] for k in keys]

    return run


_STOP = {"of", "and", "for", "the", "a", "an", "to", "in"}


def _initials(phrase: str) -> str:
    return "".join(w[0] for w in re.findall(r"[a-z0-9]+", phrase.casefold()) if w not in _STOP)


def with_acronyms(method):
    """Union clusters of an acronym phrase (single token, e.g. 'NLP',
    'LLMs') and a multi-word phrase whose initials spell it. Data-driven:
    only links acronyms whose expansion is itself in the phrase set."""

    def run(phrases, vecs, threshold):
        labels = list(method(phrases, vecs, threshold))
        parent = {l: l for l in set(labels)}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        acr = {}
        for i, p in enumerate(phrases):
            tok = p.casefold()
            if re.fullmatch(r"[a-z]{2,6}", tok):
                acr.setdefault(
                    tok.rstrip("s") if len(tok) > 3 and tok.endswith("s") else tok, []
                ).append(i)
        for j, p in enumerate(phrases):
            if len(re.findall(r"[a-z0-9]+", p.casefold())) < 2:
                continue
            ini = _initials(p)
            for i in acr.get(ini, []):
                parent[find(labels[i])] = find(labels[j])
        return [find(l) for l in labels]

    return run
