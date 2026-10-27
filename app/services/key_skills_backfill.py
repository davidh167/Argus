"""Key-skills-only Stage 2 pass for postings that never got one.

Stage 1 fast-tracks a posting straight to Stage 3, so it never goes through
Stage 2 and never gets stage2_key_skills -- which market_demand and the
interest-weighted gap analysis both read. That leaves the roles the user
hand-picked (manual captures, mostly fast-tracked) underrepresented in
exactly the data meant to describe the roles they care about.

This asks the identical Stage 2 question (stage2_batch.build_request) but
writes back ONLY stage2_key_skills: stage2_result/reason/tokens stay
untouched, so a fast-tracked posting is not retroactively "triaged" by
Stage 2 -- that would change Stage 3 eligibility and fit-tier ordering, and
skew the token averages pricing.estimate_cost relies on.

A second mode, "refresh", re-asks the question for postings that ALREADY have
key_skills and overwrites them -- used when the Stage 2 key_skills prompt
changes (v2 writes atomic, canonically spelled skills; older rows are
free-text and don't cluster with them), limited to the user's interest set
so a prompt change doesn't mean re-running thousands of bulk postings.
"""

import logging

from anthropic import Anthropic
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.posting import Posting
from app.models.verdict import Verdict
from app.services import batches as batches_service
from app.services import roadmap
from app.services.postings import latest_verdict_id_subquery
from app.services.stage2 import Stage2OutputError, Stage2UnavailableError, parse_response
from app.services.stage2_batch import _posting_id_from_custom_id, build_request

logger = logging.getLogger(__name__)

KIND = "key_skills_backfill"
KIND_REFRESH = "key_skills_refresh"


def list_pending(
    db: Session, posting_ids: list[int] | None = None, refresh: bool = False
) -> list[Posting]:
    """Open, non-dismissed postings whose latest verdict has no key_skills
    yet (or, with refresh, any key_skills) and that Stage 1 didn't skip
    outright. Scraped postings still awaiting detail hydration are left out
    -- their text is a listing blurb, not a real JD."""
    stmt = (
        select(Posting)
        .join(Verdict, Verdict.id == latest_verdict_id_subquery())
        .where(
            Verdict.stage1_result != "skip",
            Posting.dismissed_at.is_(None),
            Posting.closed_at.is_(None),
            Posting.raw_jd_text.is_not(None),
        )
    )
    if not refresh:
        stmt = stmt.where(Verdict.stage2_key_skills.is_(None))
    if posting_ids is not None:
        stmt = stmt.where(Posting.id.in_(posting_ids))
    return [
        p for p in db.scalars(stmt) if not (p.source == "scraped" and p.detail_fetched_at is None)
    ]


def _submit(db: Session, postings: list[Posting], kind: str) -> dict:
    if not postings:
        return {"batch_id": None, "submitted": 0, "capped": False, "excluded_unhydrated": 0}
    requests = [build_request(db, p) for p in postings]
    client = Anthropic(api_key=settings.anthropic_api_key)
    batch = client.messages.batches.create(requests=requests)
    batches_service.record_batch_submission(
        db,
        batch.id,
        kind=kind,
        submitted_count=len(requests),
        capped=False,
        posting_ids=[p.id for p in postings],
    )
    return {
        "batch_id": batch.id,
        "submitted": len(requests),
        "capped": False,
        "excluded_unhydrated": 0,
    }


def submit_backfill_batch(db: Session, posting_ids: list[int] | None = None) -> dict:
    if not settings.anthropic_api_key:
        raise Stage2UnavailableError("ANTHROPIC_API_KEY not configured")
    return _submit(db, list_pending(db, posting_ids), KIND)


def submit_refresh_batch(db: Session, posting_ids: list[int] | None = None) -> dict:
    """Re-extracts key_skills (current prompt) for the user's interest set --
    manual captures, starred, actively applied to -- overwriting what's there.
    posting_ids narrows it further (tests); omit for the whole interest set."""
    if not settings.anthropic_api_key:
        raise Stage2UnavailableError("ANTHROPIC_API_KEY not configured")
    postings = list_pending(db, posting_ids, refresh=True)
    weights = roadmap.interest_weights(db, postings)
    interest = [p for p in postings if weights[p.id] >= roadmap.INTEREST_SET_MIN_WEIGHT]
    return _submit(db, interest, KIND_REFRESH)


def apply_refresh_results(db: Session, batch_id: str) -> dict:
    return apply_backfill_results(db, batch_id, replace_existing=True)


def apply_backfill_results(db: Session, batch_id: str, replace_existing: bool = False) -> dict:
    """Same polling contract as stage2_batch.apply_batch_results. By default a
    result only lands if the posting's latest verdict STILL has no key_skills
    -- never overwrites ones a real Stage 2 run wrote in the meantime. A
    refresh batch (replace_existing) overwrites by design."""
    if not settings.anthropic_api_key:
        raise Stage2UnavailableError("ANTHROPIC_API_KEY not configured")

    client = Anthropic(api_key=settings.anthropic_api_key)
    batch = client.messages.batches.retrieve(batch_id)
    batches_service.update_batch_status(db, batch_id, batch.processing_status)
    if batch.processing_status != "ended":
        return {
            "status": batch.processing_status,
            "applied": 0,
            "errors": 0,
            "skipped_stale": 0,
            "posting_ids": [],
        }

    applied = errors = skipped_stale = 0
    posting_ids: list[int] = []
    for item in client.messages.batches.results(batch_id):
        posting_id = _posting_id_from_custom_id(item.custom_id)
        posting = db.get(Posting, posting_id)
        if posting is None or item.result.type in ("errored", "expired"):
            logger.error(
                "Key-skills backfill item for posting_id=%s failed: %s",
                posting_id,
                item.result.type,
            )
            errors += 1
            continue

        verdict = posting.verdicts[0]
        if verdict.stage2_key_skills is not None and not replace_existing:
            skipped_stale += 1
            continue

        try:
            _result, _reason, key_skills, _tokens = parse_response(item.result.message)
        except Stage2OutputError as exc:
            logger.error("Key-skills backfill item for posting_id=%s: %s", posting_id, exc)
            errors += 1
            continue
        verdict.stage2_key_skills = key_skills
        applied += 1
        posting_ids.append(posting_id)

    batches_service.mark_applied(db, batch_id)
    db.flush()
    return {
        "status": "ended",
        "applied": applied,
        "errors": errors,
        "skipped_stale": skipped_stale,
        "posting_ids": posting_ids,
    }
