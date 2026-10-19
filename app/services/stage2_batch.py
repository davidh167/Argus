"""Batch orchestration for Stage 2 using Anthropic's real Batch API (50%
discount vs. standard pricing). Submission and result application are two
separate steps across two separate HTTP requests -- the Batch API has no
synchronous "wait for it" call, and a batch can take up to ~an hour.
custom_id encodes posting_id directly (f"posting-{id}"); no separate
batch-tracking table, since posting_id is already the durable key needed to
look the posting/verdict back up when applying results.
"""

import logging

from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from sqlalchemy.orm import Session

from app.config import settings
from app.models.posting import Posting
from app.models.profile_version import ProfileVersion
from app.schemas.profile import ProfileSnapshotData
from app.services import batches as batches_service
from app.services import profile as profile_service
from app.services import relevance
from app.services.jd_sections import extract_jd_sections
from app.services.postings import list_stage2_pending
from app.services.stage2 import (
    STAGE2_MODEL,
    STAGE2_TOOL,
    SYSTEM_PROMPT,
    Stage2UnavailableError,
    build_user_message,
    parse_response,
)

logger = logging.getLogger(__name__)

_CUSTOM_ID_PREFIX = "posting-"


def _custom_id(posting_id: int) -> str:
    return f"{_CUSTOM_ID_PREFIX}{posting_id}"


def _posting_id_from_custom_id(custom_id: str) -> int:
    return int(custom_id.removeprefix(_CUSTOM_ID_PREFIX))


def build_request(db: Session, posting: Posting) -> Request:
    """One Stage 2 Batch API request for a posting, built from its latest
    verdict's profile version. Shared by submit_pending_batch and
    key_skills_backfill, which asks the identical question and only differs
    in what it writes back."""
    verdict = posting.verdicts[0]
    profile_version = db.get(ProfileVersion, verdict.profile_version_id)
    snapshot = ProfileSnapshotData.model_validate(profile_version.snapshot)
    sections = extract_jd_sections(posting.raw_jd_text or "")
    return Request(
        custom_id=_custom_id(posting.id),
        params=MessageCreateParamsNonStreaming(
            model=STAGE2_MODEL,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": build_user_message(
                        posting.title,
                        sections,
                        snapshot.criteria,
                        snapshot.credentials.education,
                        snapshot.skills.core,
                        snapshot.skills.developing,
                    ),
                }
            ],
            tools=[STAGE2_TOOL],
            tool_choice={"type": "tool", "name": "stage2_triage"},
        ),
    )


def submit_pending_batch(
    db: Session,
    company_id: int | None = None,
    posting_ids: list[int] | None = None,
    limit: int | None = 30,
) -> dict:
    """Builds one Batch API request per pending posting and submits them as
    a single batch. Returns immediately with the batch handle; does not
    wait for results -- see apply_batch_results. Raises
    Stage2UnavailableError if unconfigured, checked before querying so a
    misconfigured key fails fast.

    limit defaults to 30 as a safety cap -- without one, calling this with
    just a company_id submits the ENTIRE analyze-bucket backlog, which for
    a large company can be hundreds of postings. This isn't primarily a
    cost concern (even 500 postings costs well under $1 at Haiku batch
    pricing) -- it's that this endpoint runs repeatedly across every
    tracked company, and a human still eyeballs Stage 2 output, so a batch
    should stay small enough to sanity-check in one sitting. Pass
    limit=None to submit the whole pool uncapped. When capping, the pool is
    ranked by classical-NLP (TF-IDF/cosine) relevance against the current
    profile's target signals/skills -- not another LLM call, which would
    defeat the point of capping LLM spend."""
    if not settings.anthropic_api_key:
        raise Stage2UnavailableError("ANTHROPIC_API_KEY not configured")

    postings = list_stage2_pending(db, company_id=company_id, posting_ids=posting_ids)
    # A scraped posting with no detail_fetched_at yet is still just its
    # short listing blurb (Apple's average is 571 chars vs 6-14k for a
    # real JD) -- triaging that wastes the call, so it's excluded here
    # rather than in list_stage2_pending itself (which stays "everything
    # structurally eligible," used elsewhere too).
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

    capped = limit is not None and len(postings) > limit
    if capped:
        profile = ProfileSnapshotData.model_validate(
            profile_service.get_current_profile(db).snapshot
        )
        query_text = relevance.profile_query_text(profile)
        ranked = relevance.rank_by_relevance(
            [(p, p.raw_jd_text or "") for p in postings], query_text
        )
        postings = ranked[:limit]

    requests = [build_request(db, posting) for posting in postings]

    client = Anthropic(api_key=settings.anthropic_api_key)
    batch = client.messages.batches.create(requests=requests)
    batches_service.record_batch_submission(
        db,
        batch.id,
        kind="stage2_triage",
        submitted_count=len(requests),
        capped=capped,
        company_id=company_id,
        posting_ids=[p.id for p in postings],
    )
    return {
        "batch_id": batch.id,
        "submitted": len(requests),
        "capped": capped,
        "excluded_unhydrated": excluded_unhydrated,
    }


def apply_batch_results(db: Session, batch_id: str) -> dict:
    """Safe to call repeatedly (the user polls manually whenever they come
    back): if the batch hasn't ended, returns its status and applies
    nothing. Once ended, streams results and writes each succeeded one onto
    the posting's CURRENT latest verdict, looked up fresh by posting_id
    decoded from custom_id -- not a stale in-memory reference, since this
    runs in a separate request from submission, possibly after a recrawl
    created a newer verdict row for that posting. Per-item errored/expired
    results, and results whose latest verdict no longer matches the
    pending-Stage-2 shape, are counted and reported, not raised -- partial
    failure across independent items is expected, not exceptional.

    posting_ids in the return lists postings this call actually wrote a
    fresh stage2_result onto -- lets a caller link directly to results
    instead of only reporting a count."""
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
        if posting is None:
            logger.error("Batch result for unknown posting_id=%s", posting_id)
            errors += 1
            continue

        outcome = item.result
        if outcome.type in ("errored", "expired"):
            logger.error(
                "Stage 2 batch item %s for posting_id=%s: %s",
                outcome.type,
                posting_id,
                getattr(outcome, "error", None),
            )
            errors += 1
            continue

        verdict = posting.verdicts[0]
        if verdict.stage1_result != "analyze" or verdict.stage2_result is not None:
            # Latest verdict changed since submission (e.g. a recrawl ran
            # while the batch was in flight) -- don't write a Stage 2
            # result onto a verdict row it was never computed for.
            skipped_stale += 1
            continue

        result, reason, key_skills, tokens = parse_response(outcome.message)
        verdict.stage2_result = result
        verdict.stage2_reason = reason
        verdict.stage2_key_skills = key_skills
        verdict.stage2_tokens_used = tokens
        # Correct even though we don't re-derive it from the original
        # request: the guard just above proved this verdict is still the
        # exact pending-Stage-2 row it was submitted for, so its
        # profile_version_id is unchanged from submit time -- the same
        # value submit_pending_batch used to build the criteria.
        verdict.stage2_profile_version_id = verdict.profile_version_id
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
