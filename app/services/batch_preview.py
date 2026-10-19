"""Read-only cost/count previews for the Stage 2/3 batch submit endpoints
(app.services.stage2_batch/stage3_batch) -- mirrors their exact
eligibility math (list_stage2_pending/list_stage3_pending, the same
unhydrated-scraped exclusion, the same capping arithmetic) without
calling the Anthropic API or the relevance ranker: a preview only needs
the COUNT that would be submitted, not which specific postings the
relevance-based cap would pick. Never mutates anything, safe to call on
every filter change.
"""

from sqlalchemy.orm import Session

from app.models.verdict import Verdict
from app.services import pricing
from app.services.postings import list_stage2_pending, list_stage3_pending


def _eligible_after_hydration_exclusion(postings: list) -> tuple[list, int]:
    excluded_unhydrated = sum(
        1 for p in postings if p.source == "scraped" and p.detail_fetched_at is None
    )
    eligible = [p for p in postings if not (p.source == "scraped" and p.detail_fetched_at is None)]
    return eligible, excluded_unhydrated


def _preview(
    eligible: list, excluded_unhydrated: int, limit: int | None, in_flight: int = 0
) -> dict:
    capped = limit is not None and len(eligible) > limit
    would_submit = min(len(eligible), limit) if limit is not None else len(eligible)
    return {
        "eligible": len(eligible),
        "would_submit": would_submit,
        "capped": capped,
        "excluded_unhydrated": excluded_unhydrated,
        "in_pending_batches": in_flight,
    }


def preview_stage2(
    db: Session,
    company_id: int | None = None,
    posting_ids: list[int] | None = None,
    limit: int | None = 30,
) -> dict:
    postings = list_stage2_pending(db, company_id=company_id, posting_ids=posting_ids)
    in_flight = len(
        list_stage2_pending(
            db, company_id=company_id, posting_ids=posting_ids, exclude_in_flight=False
        )
    ) - len(postings)
    eligible, excluded_unhydrated = _eligible_after_hydration_exclusion(postings)
    result = _preview(eligible, excluded_unhydrated, limit, in_flight)
    result.update(
        pricing.estimate_cost(
            db,
            Verdict.stage2_tokens_used,
            result["would_submit"],
            pricing.HAIKU_BATCH_PRICE_PER_MTOK,
        )
    )
    return result


def preview_stage3(
    db: Session,
    company_id: int | None = None,
    posting_ids: list[int] | None = None,
    limit: int | None = 30,
    include_stale: bool = False,
    include_unready: bool = False,
) -> dict:
    kwargs = dict(
        company_id=company_id,
        posting_ids=posting_ids,
        include_stale=include_stale,
        include_unready=include_unready,
    )
    postings = list_stage3_pending(db, **kwargs)
    in_flight = len(list_stage3_pending(db, exclude_in_flight=False, **kwargs)) - len(postings)
    eligible, excluded_unhydrated = _eligible_after_hydration_exclusion(postings)
    result = _preview(eligible, excluded_unhydrated, limit, in_flight)
    result.update(
        pricing.estimate_cost(
            db,
            Verdict.stage3_tokens_used,
            result["would_submit"],
            pricing.SONNET_BATCH_PRICE_PER_MTOK,
        )
    )
    return result
