"""Recompute compensation for every posting whose raw_jd_text hasn't been
run through the extractor yet (comp_source is null) -- pure local
recomputation over already-stored text, no network, no LLM, safe to run
any time (e.g. after compensation.py's patterns are improved, or after a
re-crawl brings in cleaner text -- see clean_html_text's Anthropic fix).
Never overwrites an ATS-structured value (comp_source='ats_field') with a
worse text-derived guess.

Usage:
    poetry run python scripts/backfill_compensation.py [--company NAME] [--dry-run]

Prints per-company coverage (before -> after) when done.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from app.db import SessionLocal
from app.models.company import Company
from app.models.posting import Posting
from app.services import compensation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--company", default=None, help="Only backfill postings for this company name"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would change, write nothing"
    )
    args = parser.parse_args()

    with SessionLocal() as db:
        stmt = (
            select(Posting)
            .join(Company)
            .where(Posting.comp_source.is_(None), Posting.raw_jd_text.is_not(None))
        )
        if args.company:
            stmt = stmt.where(Company.name == args.company)
        candidates = list(db.scalars(stmt))

        before_by_company = Counter()
        after_by_company = Counter()
        total_by_company = Counter()

        # Coverage denominator is every posting for the touched companies,
        # not just the candidates, so "before" reflects reality (including
        # postings that already had ats_field comp from a crawl).
        touched_company_ids = {p.company_id for p in candidates}
        if touched_company_ids:
            all_postings = list(
                db.scalars(select(Posting).where(Posting.company_id.in_(touched_company_ids)))
            )
            for p in all_postings:
                total_by_company[p.company.name] += 1
                if p.comp_source is not None:
                    before_by_company[p.company.name] += 1
                    after_by_company[p.company.name] += 1

        updated = 0
        for posting in candidates:
            extracted = compensation.extract_compensation(posting.raw_jd_text)
            if extracted is None:
                continue
            updated += 1
            after_by_company[posting.company.name] += 1
            if not args.dry_run:
                posting.comp_min = extracted.comp_min
                posting.comp_max = extracted.comp_max
                posting.comp_currency = extracted.comp_currency
                posting.comp_period = extracted.comp_period
                posting.comp_source = "jd_text"
                posting.comp_raw = extracted.comp_raw

        if not args.dry_run:
            db.commit()

        print(
            f"{'[dry run] ' if args.dry_run else ''}Scanned {len(candidates)} postings without comp_source."
        )
        print(f"Extracted comp for {updated} of them.\n")
        print("Coverage by company (had comp / total postings):")
        for name in sorted(total_by_company, key=lambda n: -total_by_company[n]):
            print(
                f"  {name:20s} {before_by_company[name]:5d} -> {after_by_company[name]:5d} "
                f"/ {total_by_company[name]:5d}"
            )


if __name__ == "__main__":
    main()
