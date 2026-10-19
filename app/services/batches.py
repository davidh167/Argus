"""Tracks Anthropic Batch API submissions (Stage 2 triage, Stage 3
analysis) so any interface -- curl, MCP, extension, desktop -- can see
what's pending and survives a desktop app restart. Persisted here rather
than left as UI-local state, which is exactly the gap that motivated this
module: a batch submitted outside the desktop app, or before a reload,
was previously invisible to it.
"""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.batch_submission import BatchSubmission


def record_batch_submission(
    db: Session,
    batch_id: str,
    kind: str,
    submitted_count: int,
    capped: bool,
    company_id: int | None = None,
    profile_version_id: int | None = None,
    posting_ids: list[int] | None = None,
) -> BatchSubmission:
    submission = BatchSubmission(
        batch_id=batch_id,
        kind=kind,
        submitted_count=submitted_count,
        capped=capped,
        company_id=company_id,
        profile_version_id=profile_version_id,
        posting_ids=posting_ids,
    )
    db.add(submission)
    db.flush()
    return submission


def get_batch_submission(db: Session, batch_id: str) -> BatchSubmission | None:
    return db.scalar(select(BatchSubmission).where(BatchSubmission.batch_id == batch_id))


def update_batch_status(db: Session, batch_id: str, status: str) -> None:
    """No-ops silently if batch_id isn't tracked -- shouldn't happen since
    every submission path records one, but a missing row here isn't worth
    raising over; it just means the cache doesn't get updated."""
    submission = db.scalar(select(BatchSubmission).where(BatchSubmission.batch_id == batch_id))
    if submission is None:
        return
    submission.last_status = status
    submission.last_checked_at = datetime.now(timezone.utc)
    db.flush()


def mark_applied(db: Session, batch_id: str) -> None:
    """Called once a batch's results have been applied (it had ended)."""
    submission = db.scalar(select(BatchSubmission).where(BatchSubmission.batch_id == batch_id))
    if submission is not None and submission.applied_at is None:
        submission.applied_at = datetime.now(timezone.utc)
        db.flush()


def in_flight_posting_ids(db: Session, kind: str) -> set[int]:
    """Postings in a submitted batch of this kind whose results haven't been
    applied yet -- excluded from that stage's eligibility so they aren't
    submitted (and paid for) twice."""
    rows = db.scalars(
        select(BatchSubmission.posting_ids).where(
            BatchSubmission.kind == kind,
            BatchSubmission.applied_at.is_(None),
            BatchSubmission.posting_ids.is_not(None),
        )
    )
    return {pid for ids in rows for pid in ids}


def list_batch_submissions(db: Session, kind: str | None = None) -> list[BatchSubmission]:
    stmt = select(BatchSubmission)
    if kind is not None:
        stmt = stmt.where(BatchSubmission.kind == kind)
    # id as a tiebreaker: Postgres's now() returns the same value for every
    # call within one transaction, so created_at alone can't distinguish
    # rows inserted back-to-back in the same transaction (as happens in
    # tests, and in any single request that submits and records at once).
    stmt = stmt.order_by(BatchSubmission.created_at.desc(), BatchSubmission.id.desc())
    return list(db.scalars(stmt))
