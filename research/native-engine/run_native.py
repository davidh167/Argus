"""Phase 2 of docs/design/native_engine_plan.md: native sparse TF-IDF similarity vs the Python implementation.

Loads each corpus into the throwaway Postgres database (`<configured db>_bench`, as run_baseline.py does),
checks parity on a seeded sample of postings, then times find_similar_postings under both engines plus the
engine alone, and measures index memory in fresh subprocesses. Needs the built libargus (native/README.md).
No Ollama, no model, no API calls.

Run from the project root:

  PYTHONPATH=. poetry run python research/native-engine/run_native.py \\
      --synthetic 3000 --out research/native-engine/reports/native_synthetic_3000.md
  PYTHONPATH=. poetry run python research/native-engine/run_native.py \\
      --synthetic 30000 --runs 10 --warmup 1 --out research/native-engine/reports/native_synthetic_30000.md
  PYTHONPATH=. poetry run python research/native-engine/run_native.py \\
      --snapshot research/ranking-eval/data --out research/native-engine/reports/native_snapshot.md
"""

import argparse
import sys
from pathlib import Path

from bench_corpus import Corpus, generate_synthetic, load_snapshot
from bench_db import throwaway_database


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--synthetic", type=int, action="append", default=[], metavar="N")
    parser.add_argument("--snapshot", type=Path, metavar="DIR")
    parser.add_argument("--runs", type=int, default=30, help="timed runs per case (>= 10)")
    parser.add_argument(
        "--query-runs", type=int, default=200, help="timed runs of the fast native query cases"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--parity-sample", type=int, default=100, help="query postings in the parity gate"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--db-name")
    args = parser.parse_args()
    if args.runs < 10:
        parser.error("--runs must be at least 10")
    if not args.synthetic and args.snapshot is None:
        parser.error("give at least one of --synthetic N, --snapshot DIR")

    corpora: list[tuple[Corpus, list[str]]] = [
        (generate_synthetic(n), ["--n", str(n)]) for n in args.synthetic
    ]
    if args.snapshot is not None:
        corpora.append((load_snapshot(args.snapshot), ["--snapshot", str(args.snapshot)]))

    with throwaway_database(args.db_name) as db_name:
        # Imported only now: app.db builds its engine on import, and must see the throwaway URL.
        from sqlalchemy import text

        import bench_native
        import bench_report
        from app.db import SessionLocal
        from app.services import relevance
        from bench_db import clear_corpus, load_corpus

        with SessionLocal() as db:
            pg_version = str(db.scalar(text("select version()"))).split(" on ")[0]

        sections = []
        for corpus, memory_args in corpora:
            n = len(corpus.postings)
            print(f"== {corpus.name}: loading into {db_name}", file=sys.stderr)
            with SessionLocal() as db:
                load_corpus(db, corpus)
                db.commit()
            relevance._similarity_index_cache.clear()
            documents = bench_native.fetch_documents(SessionLocal)
            print(f"== {corpus.name}: parity gate", file=sys.stderr)
            parity = bench_native.parity_gate(documents, args.parity_sample)
            if not parity.ok:
                raise SystemExit(
                    f"parity gate failed on {corpus.name}: {len(parity.mismatches)} mismatches"
                )
            print(f"== {corpus.name}: timing", file=sys.stderr)
            timings = bench_native.run_timings(
                SessionLocal,
                documents,
                runs=args.runs,
                warmup=args.warmup,
                query_runs=args.query_runs,
            )
            print(f"== {corpus.name}: memory", file=sys.stderr)
            memory = bench_native.memory_runs(memory_args)
            sections.append(
                bench_native.corpus_section(corpus.name, n, timings, memory, parity, corpus.notes)
            )
            with SessionLocal() as db:
                clear_corpus(db)

    report = bench_native.render_report(bench_report.environment(pg_version), sections)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
