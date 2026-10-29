"""Batch orchestration for Stage 3 using Anthropic's real Batch API (50%
discount vs. standard pricing). Mirrors app/services/stage2_batch.py
closely -- same submit/apply split across two separate HTTP requests (no
synchronous "wait for it" call, a batch can take up to ~an hour), same
custom_id-encodes-posting_id pattern, same relevance-ranked capping. The
real differences: Stage 3 has no stage1/stage2 precondition (manually
triggered), uses Sonnet not Haiku, and needs the current
profile + in_progress roadmap items threaded into every request, same as
run_stage3_analysis does for the synchronous single-posting path.
"""

import logging
from datetime import datetime, timezone

from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from sqlalchemy.orm import Session

from app.config import settings
from app.models.posting import Posting
from app.schemas.profile import ProfileSnapshotData
from app.services import batches as batches_service
from app.services import profile as profile_service
from app.services import relevance, stage3
from app.services import roadmap as roadmap_service
from app.services.jd_sections import extract_jd_sections
from app.services.postings import list_stage3_pending
from app.services.stage2 import Stage2UnavailableError
from app.services.stage3 import STAGE3_MODEL, STAGE3_TOOL, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_CUSTOM_ID_PREFIX = "posting-"


def _custom_id(posting_id: int) -> str:
    return f"{_CUSTOM_ID_PREFIX}{posting_id}"


def _posting_id_from_custom_id(custom_id: str) -> int:
    return int(custom_id.removeprefix(_CUSTOM_ID_PREFIX))


def submit_pending_batch(
    db: Session,
    company_id: int | None = None,
    posting_ids: list[int] | None = None,
    limit: int | None = 30,
    include_stale: bool = False,
    include_unready: bool = False,
) -> dict:
    """Builds one Batch API request per pending posting and submits them as
    a single batch. Returns immediately with the batch handle; does not
    wait for results -- see apply_batch_results. Raises
    Stage2UnavailableError if unconfigured, checked before querying so a
    misconfigured key fails fast.

    limit defaults to 30, same reasoning as Stage 2's batch cap: a human
    still eyeballs Stage 3 output, so a batch should stay small enough to
    sanity-check in one sitting -- not primarily a cost concern (Sonnet
    batch pricing for 30 postings is still well under a dollar). Pass
    limit=None to submit the whole pool uncapped. When capping, the pool is
    ranked by classical-NLP (TF-IDF/cosine) relevance against the current
    profile, same as Stage 2's cap -- not another LLM call.

    include_stale=True additionally includes postings whose Stage 3 result
    is stale (4.4's roadmap -> profile -> re-fit loop: after adding a
    completed roadmap item's skill to the profile, the evidence postings
    that surfaced it become stale and this is how the user deliberately
    re-runs them). Never the default -- see list_stage3_pending for why."""
    if not settings.anthropic_api_key:
        raise Stage2UnavailableError("ANTHROPIC_API_KEY not configured")

    postings = list_stage3_pending(
        db,
        company_id=company_id,
        posting_ids=posting_ids,
        include_stale=include_stale,
        include_unready=include_unready,
    )
    # Same reasoning as Stage 2: an unhydrated scraped posting is still
    # just its short listing blurb, not worth a Sonnet call.
    excluded_unhydrated = sum(
        1 for p in postings if p.source == "scraped" and p.detail_fetched_at is None
    )
    postings = [p for p in postings if not (p.source == "scraped" and p.detail_fetched_at is None)]
    if not postings:
        return {
            "batch_id": None,
            "submitted": 0,
            "capped": False,
            "excluded_unhydrated": excluded_unhydrated,
        }

    profile_version = profile_service.get_current_profile(db)
    profile = ProfileSnapshotData.model_validate(profile_version.snapshot)

    capped = limit is not None and len(postings) > limit
    if capped:
        query_text = relevance.profile_query_text(profile)
        ranked = relevance.rank_by_relevance(
            [(p, p.raw_jd_text or "") for p in postings], query_text
        )
        postings = ranked[:limit]

    roadmap_items = roadmap_service.get_in_progress_skills(db)

    requests: list[Request] = []
    for posting in postings:
        sections = extract_jd_sections(posting.raw_jd_text or "")
        requests.append(
            Request(
                custom_id=_custom_id(posting.id),
                params=MessageCreateParamsNonStreaming(
                    model=STAGE3_MODEL,
                    max_tokens=2048,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": stage3.build_user_message(
                                title=posting.title,
                                sections=sections,
                                profile=profile,
                                roadmap_items=roadmap_items,
                            ),
                        }
                    ],
                    tools=[STAGE3_TOOL],
                    tool_choice={"type": "tool", "name": "stage3_fit_analysis"},
                ),
            )
        )

    client = Anthropic(api_key=settings.anthropic_api_key)
    batch = client.messages.batches.create(requests=requests)
    batches_service.record_batch_submission(
        db,
        batch.id,
        kind="stage3_analysis",
        submitted_count=len(requests),
        capped=capped,
        company_id=company_id,
        profile_version_id=profile_version.id,
        posting_ids=[p.id for p in postings],
    )
    return {
        "batch_id": batch.id,
        "submitted": len(requests),
        "capped": capped,
        "excluded_unhydrated": excluded_unhydrated,
    }


def apply_batch_results(db: Session, batch_id: str) -> dict:
    """Safe to call repeatedly: if the batch hasn't ended, returns its
    status and applies nothing. Once ended, streams results and writes each
    succeeded one onto the posting's CURRENT latest verdict, looked up
    fresh by posting_id decoded from custom_id. Per-item errored/expired
    results, and results whose latest verdict already has a stage3_result
    (analyzed via another path while the batch was in flight), are counted
    and reported, not raised.

    posting_ids in the return is every posting with a real, viewable
    verdict as of this call (freshly applied this call, or already there
    from a prior call/path) -- lets a caller show/link to results directly
    instead of only reporting counts. Stable across repeat calls on an
    already-ended batch: Anthropic keeps batch results available, so
    re-checking a long-done batch still returns the same posting_ids."""
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

    # The profile version every request in this batch was built against,
    # recorded at submit time (see submit_pending_batch) -- unlike Stage 2,
    # this can't be derived from the verdict itself at apply time, since
    # apply can happen long after submission, possibly after the profile
    # changed again. None if this batch_id was never tracked (e.g.
    # submitted before this field existed) -- results still apply, they
    # just can't be marked stale later.
    tracked_submission = batches_service.get_batch_submission(db, batch_id)
    submitted_profile_version_id = (
        tracked_submission.profile_version_id if tracked_submission else None
    )

    applied = errors = skipped_stale = 0
    posting_ids: list[int] = []
    for item in client.messages.batches.results(batch_id):
        posting_id = _posting_id_from_custom_id(item.custom_id)
        posting = db.get(Posting, posting_id)
        if posting is None:
            logger.error("Batch result for unknown posting_id=%s", posting_id)
            errors += 1
            continue

        outcome = item.result
        if outcome.type != "succeeded":
            logger.error(
                "Stage 3 batch item %s for posting_id=%s: %s",
                outcome.type,
                posting_id,
                getattr(outcome, "error", None),
            )
            errors += 1
            continue

        verdict = posting.verdicts[0]
        if verdict.stage3_result is not None:
            skipped_stale += 1
            posting_ids.append(posting_id)
            continue

        result, tokens = stage3.parse_response(outcome.message)
        verdict.stage3_triggered_at = datetime.now(timezone.utc)
        verdict.stage3_result = result.model_dump(mode="json")
        verdict.stage3_tokens_used = tokens
        verdict.stage3_profile_version_id = submitted_profile_version_id
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
