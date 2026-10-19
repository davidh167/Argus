"""Recompute normalized_jd_text for every posting that has raw_jd_text but
no normalized_jd_text yet -- pure local recomputation, no network, no LLM.
Safe to re-run any time (e.g. after jd_sections.py's heuristics improve, or
after a re-crawl brings in cleaner raw_jd_text -- see clean_html_text's
Anthropic fix) since it always recomputes from the CURRENT raw_jd_text.

Usage:
    poetry run python scripts/backfill_normalized_jd_text.py [--dry-run]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from app.db import SessionLocal
from app.models.posting import Posting
from app.services.jd_sections import build_normalized_jd_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with SessionLocal() as db:
        candidates = list(
            db.scalars(
                select(Posting).where(
                    Posting.normalized_jd_text.is_(None), Posting.raw_jd_text.is_not(None)
                )
            )
        )
        updated = 0
        for posting in candidates:
            normalized = build_normalized_jd_text(posting.raw_jd_text)
            if normalized is None:
                continue
            updated += 1
            if not args.dry_run:
                posting.normalized_jd_text = normalized

        if not args.dry_run:
            db.commit()

        print(
            f"{'[dry run] ' if args.dry_run else ''}Scanned {len(candidates)} postings without normalized_jd_text."
        )
        print(f"Populated {updated} of them.")


if __name__ == "__main__":
    main()
