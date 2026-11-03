"""Exact-parity check of the native similarity engine on a real exported snapshot (Phase 2).

Loads the snapshot (research/ranking-eval/export_snapshot.py -> research/ranking-eval/data/) with the Phase 0
loader into the throwaway database `<configured db>_bench` (created and dropped here; the configured database is
never opened), reads the postings back the way the app does, and for EVERY posting compares the native engine's
top-10 with the Python implementation's: same ids, same order, scores within 1e-9. Writes a markdown report with
the counts and the first mismatches. Needs the built libargus (native/README.md). Run by the owner:

  PYTHONPATH=. poetry run python research/native-engine/parity_snapshot.py \\
      --snapshot research/ranking-eval/data --out research/native-engine/reports/parity_snapshot.md

The Python reference is the pure-Python cosine loop, so expect a few minutes at ~3,400 postings (it prints
progress). `--synthetic N` runs the same thing on the seeded synthetic corpus (how this script is exercised
without the real data); `--sample K` checks a seeded sample of K query postings instead of all.
"""

import argparse
import platform
import sys
import time
import unicodedata
from pathlib import Path

from bench_corpus import Corpus, generate_synthetic, load_snapshot
from bench_db import throwaway_database

EXAMPLES = 10


def render(
    corpus: Corpus,
    documents: list[tuple[int, str]],
    result,  # bench_parity.ParityResult
    seconds: float,
    versions: dict[str, str],
    n_non_ascii: int,
) -> str:
    ok = result.ok
    lines = [
        "# Native similarity parity on a corpus (Phase 2)",
        "",
        f"- corpus: {corpus.name}, {len(documents):,} postings ({n_non_ascii:,} with non-ASCII text)",
        f"- query postings checked: {result.n_targets:,} (top-{result.k} each, {result.n_hits:,} hits)",
        f"- machine: {platform.platform()}, Python {platform.python_version()}",
        "- " + "\n- ".join(f"{k}: {v}" for k, v in versions.items()),
        f"- wall time: {seconds:.0f} s",
        "",
        f"**Result: {'PASS' if ok else 'FAIL'}.**",
        "",
        "| Measure | Count |",
        "|---|---:|",
        f"| query postings | {result.n_targets:,} |",
        f"| top-{result.k} lists with different ids or order | {result.n_id_mismatch:,} |",
        f"|   of which only a reordering of scores within 1e-9 | {result.n_near_tie_reorders:,} |",
        f"| lists with identical ids but a score off by more than 1e-9 | {result.n_score_mismatch:,} |",
        f"| max absolute score difference over all identical-id lists | {result.max_abs_score_diff:.3g} |",
        f"| scores bit-identical to Python's | {result.n_exact_score_matches:,} of {result.n_hits:,} |",
        "",
        "A list passes when the ids match in order and every score is within 1e-9. Ties (equal scores) are "
        "broken by corpus order in both engines; a 'reordering within 1e-9' means two documents whose scores "
        "differ by floating-point noise swapped places.",
    ]
    if result.mismatches:
        lines += ["", f"## First {min(EXAMPLES, len(result.mismatches))} mismatches", ""]
        for m in result.mismatches[:EXAMPLES]:
            lines += [
                f"- posting {m.posting_id} ({m.kind}):",
                f"  - python: {[(i, round(x, 12)) for i, x in m.python]}",
                f"  - native: {[(i, round(x, 12)) for i, x in m.native]}",
            ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--snapshot", type=Path, metavar="DIR")
    parser.add_argument("--synthetic", type=int, metavar="N")
    parser.add_argument(
        "--sample", type=int, metavar="K", help="check K seeded query postings, not all"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--db-name")
    args = parser.parse_args()
    if (args.snapshot is None) == (args.synthetic is None):
        parser.error("give exactly one of --snapshot DIR, --synthetic N")

    corpus = load_snapshot(args.snapshot) if args.snapshot else generate_synthetic(args.synthetic)

    with throwaway_database(args.db_name) as db_name:
        # Imported only now: app.db builds its engine on import, and must see the throwaway URL.
        import bench_native
        import bench_parity
        from app import native
        from app.db import SessionLocal
        from bench_db import clear_corpus, load_corpus

        print(f"== {corpus.name}: loading into {db_name}", file=sys.stderr)
        with SessionLocal() as db:
            load_corpus(db, corpus)
            db.commit()
        documents = bench_native.fetch_documents(SessionLocal)
        with SessionLocal() as db:
            clear_corpus(db)

    def progress(done: int, total: int) -> None:
        if done % 200 == 0 or done == total:
            print(f"  {done}/{total} postings compared", file=sys.stderr, flush=True)

    print(f"== comparing {len(documents):,} postings", file=sys.stderr)
    start = time.perf_counter()
    result = bench_parity.compare(documents, k=10, sample=args.sample, progress=progress)
    seconds = time.perf_counter() - start

    unicode_version, icu_version = native.unicode_versions()
    versions = {
        "Python Unicode database": unicodedata.unidata_version,
        "ICU": f"{icu_version} (Unicode {unicode_version})",
        "native library": str(native.lib_path()),
    }
    n_non_ascii = sum(1 for _, text in documents if not text.isascii())
    report = render(corpus, documents, result, seconds, versions, n_non_ascii)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"wrote {args.out}: {'PASS' if result.ok else 'FAIL'}", file=sys.stderr)
    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
