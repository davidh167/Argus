"""Loops POST /postings/hydrate's underlying service until every
source='scraped' posting has a detail_fetched_at, or --max-batches runs
out. THIS SCRIPT MAKES REAL NETWORK REQUESTS (one HTTP fetch per posting,
~1 req/s) -- it is meant to be run by you, not by an agent on your behalf.

Usage:
    poetry run python scripts/backfill_hydration.py [--company NAME] [--batch-size 25] [--max-batches 200]

Prints progress after each batch (hydrated / failed / remaining).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import SessionLocal
from app.models.company import Company
from app.services.postings import hydrate_scraped_postings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--company", default=None, help="Only hydrate postings for this company name"
    )
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument(
        "--max-batches", type=int, default=200, help="Safety cap on how many batches to run"
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=1.0,
        help="Pause between batches, polite to the target site",
    )
    args = parser.parse_args()

    with SessionLocal() as db:
        company_id = None
        if args.company:
            company = db.query(Company).filter(Company.name == args.company).first()
            if company is None:
                print(f"No company named {args.company!r}")
                return
            company_id = company.id

        total_hydrated = total_failed = 0
        for batch_num in range(1, args.max_batches + 1):
            result = hydrate_scraped_postings(db, company_id=company_id, limit=args.batch_size)
            db.commit()
            total_hydrated += result["hydrated"]
            total_failed += result["failed"]
            print(
                f"Batch {batch_num}: hydrated={result['hydrated']} failed={result['failed']} "
                f"remaining={result['remaining']} (totals: hydrated={total_hydrated}, failed={total_failed})"
            )
            if result["remaining"] == 0:
                print("Done -- no postings left to hydrate.")
                return
            if result["hydrated"] == 0 and result["failed"] == 0:
                print("Nothing happened this batch -- stopping to avoid looping forever.")
                return
            time.sleep(args.pause_seconds)

        print(f"Stopped after {args.max_batches} batches (--max-batches). Re-run to continue.")


if __name__ == "__main__":
    main()
