"""Index memory of one engine in a fresh process (run by bench_native.memory_runs).

Prints one JSON object: resident-set size (MB, from `ps -o rss`) before building the index, after building it
and a full garbage collection, and the difference. The corpus is the seeded synthetic one (or the snapshot), its
(id, similarity text) pairs already in memory before the baseline is taken.
"""

import argparse
import gc
import json
import os
import subprocess
from pathlib import Path

from bench_corpus import generate_synthetic, load_snapshot


def rss_mb() -> float:
    kib = int(
        subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=True
        ).stdout
    )
    return kib / 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["python", "native"], required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--n", type=int, help="synthetic corpus of this size")
    source.add_argument("--snapshot", type=Path)
    args = parser.parse_args()

    from app import native
    from app.services import relevance

    corpus = load_snapshot(args.snapshot) if args.snapshot else generate_synthetic(args.n)
    documents = [(p.id, relevance._similarity_text(p.title, p.text)) for p in corpus.postings]
    del corpus
    native.tokenize("warm")  # loads the library and touches ICU's data before the baseline
    gc.collect()
    before = rss_mb()
    index = (
        native.Similarity(documents)
        if args.engine == "native"
        else relevance.python_similarity_vectors(documents)
    )
    gc.collect()
    after = rss_mb()
    print(
        json.dumps(
            {
                "engine": args.engine,
                "n": len(documents),
                "before_mb": before,
                "after_mb": after,
                "delta_mb": after - before,
            }
        )
    )
    del index


if __name__ == "__main__":
    main()
