"""Phase 0 of docs/design/native_engine_plan.md: where does the time go in the current Python code?

Times relevance.find_similar_postings (cold and warm), relevance.rescore_all, relevance.bm25_scores and
roadmap.market_demand (stubbed embedder) on a corpus loaded into a THROWAWAY Postgres database that
this script creates (`<configured db>_bench`) and drops; the configured database is never opened.
Also scans the corpus for non-ASCII tokens (decision D4). No Ollama, no native library, no API calls.

Run from the project root (see research/native-engine/README.md):

  PYTHONPATH=. poetry run python research/native-engine/run_baseline.py \\
      --synthetic 3000 --synthetic 30000 --out research/native-engine/reports/baseline_synthetic.md
  PYTHONPATH=. poetry run python research/native-engine/run_baseline.py \\
      --snapshot research/ranking-eval/data --out research/native-engine/reports/baseline_snapshot.md
"""

import argparse
import logging
import sys
from pathlib import Path

from bench_corpus import Corpus, generate_synthetic, load_snapshot
from bench_db import throwaway_database


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--synthetic", type=int, action="append", default=[], metavar="N")
    parser.add_argument(
        "--snapshot", type=Path, metavar="DIR", help="dir with snapshot.jsonl + meta.json"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="timed runs per case (>= 10; use >= 30 unless a case is too slow at large N)",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--db-name", help="throwaway database name (default: <configured>_bench)")
    args = parser.parse_args()
    if args.runs < 10:
        parser.error("--runs must be at least 10")
    if not args.synthetic and args.snapshot is None:
        parser.error("give at least one of --synthetic N, --snapshot DIR")

    corpora: list[Corpus] = [generate_synthetic(n) for n in args.synthetic]
    if args.snapshot is not None:
        corpora.append(load_snapshot(args.snapshot))

    with throwaway_database(args.db_name) as db_name:
        # Imported only now: app.db builds its engine on import, and must see the throwaway URL.
        from sqlalchemy import text

        import bench_cases
        import bench_report
        from app.db import SessionLocal, engine
        from app.services import relevance
        from bench_db import clear_corpus, load_corpus
        from bench_scan import scan

        assert engine.url.database == db_name
        # Alembic's logging config (run by the migration) leaves bm25s logging at DEBUG per index build.
        logging.getLogger("bm25s").setLevel(logging.WARNING)
        with SessionLocal() as db:
            pg_version = str(db.scalar(text("select version()"))).split(" on ")[0]

        sections = []
        for corpus in corpora:
            print(f"== {corpus.name}: loading into {db_name}", file=sys.stderr)
            with SessionLocal() as db:
                load_corpus(db, corpus)
                db.commit()
            relevance._similarity_index_cache.clear()
            print(f"== {corpus.name}: timing cases", file=sys.stderr)
            results = bench_cases.run_cases(SessionLocal, runs=args.runs, warmup=args.warmup)
            print(f"== {corpus.name}: non-ASCII scan", file=sys.stderr)
            scan_result = scan([(p.title, p.text) for p in corpus.postings], progress=print)
            sections.append(
                bench_report.corpus_section(
                    corpus.name, len(corpus.postings), results, scan_result, corpus.notes
                )
            )
            with SessionLocal() as db:
                clear_corpus(db)

    report = bench_report.render_report(bench_report.environment(pg_version), sections)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
