from collections.abc import Callable
from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas.posting import (
    AnalyzePendingApplied,
    AnalyzePendingSubmitted,
    BatchPreview,
    CapturePostingRequest,
    HydrateResult,
    JDSectionsOut,
    PostingCreated,
    PostingDetail,
    PostingSummary,
    RescoreResult,
    SimilarPosting,
    SpendSummary,
    TriagePendingApplied,
    TriagePendingSubmitted,
)
from app.schemas.verdict import VerdictOut, build_verdict_out
from app.services import batch_preview as batch_preview_service
from app.services import batches as batches_service
from app.services import key_skills_backfill, stage2_batch, stage3_batch
from app.services import postings as postings_service
from app.services import pricing as pricing_service
from app.services import profile as profile_service
from app.services import relevance as relevance_service
from app.services import stage3 as stage3_service
from app.services.jd_sections import extract_jd_sections
from app.utils.freshness import freshness_flag

router = APIRouter(prefix="/postings", tags=["postings"])


def _to_summary(posting) -> PostingSummary:
    return PostingSummary(
        id=posting.id,
        company_id=posting.company_id,
        company_name=posting.company.name,
        title=posting.title,
        url=posting.url,
        source=posting.source,
        starred=posting.starred,
        starred_at=posting.starred_at,
        dismissed=posting.dismissed_at is not None,
        dismissed_at=posting.dismissed_at,
        note=posting.note,
        freshness=freshness_flag(posting.posted_date, posting.first_seen_at, posting.closed_at),
        repost_count=posting.repost_count,
        stage1_result=posting.verdicts[0].stage1_result if posting.verdicts else None,
        stage2_result=posting.verdicts[0].stage2_result if posting.verdicts else None,
        stage3_verdict=(
            (posting.verdicts[0].stage3_result or {}).get("verdict")
            if posting.verdicts and posting.verdicts[0].stage3_result
            else None
        ),
        locations=posting.locations,
        workplace_type=posting.workplace_type,
        comp_min=posting.comp_min,
        comp_max=posting.comp_max,
        comp_currency=posting.comp_currency,
        comp_period=posting.comp_period,
        relevance_score=posting.relevance_score,
    )


def _to_detail(posting, current_profile_version_id: int) -> PostingDetail:
    summary = _to_summary(posting)
    latest_verdict = (
        build_verdict_out(posting.verdicts[0], current_profile_version_id)
        if posting.verdicts
        else None
    )
    jd_sections = None
    if posting.raw_jd_text:
        sections = extract_jd_sections(posting.raw_jd_text, max_chars=None)
        jd_sections = JDSectionsOut(
            summary=sections.summary,
            responsibilities=sections.responsibilities,
            required_qualifications=sections.required_qualifications,
            preferred_qualifications=sections.preferred_qualifications,
        )
    return PostingDetail(
        **summary.model_dump(),
        raw_jd_text=posting.raw_jd_text,
        posted_date=posting.posted_date,
        first_seen_at=posting.first_seen_at,
        last_seen_at=posting.last_seen_at,
        closed_at=posting.closed_at,
        department=posting.department,
        comp_source=posting.comp_source,
        comp_raw=posting.comp_raw,
        detail_fetched_at=posting.detail_fetched_at,
        relevance_matched_terms=posting.relevance_matched_terms,
        jd_sections=jd_sections,
        latest_verdict=latest_verdict,
    )


@router.post("/capture", response_model=PostingCreated)
def capture_posting(payload: CapturePostingRequest, db: Session = Depends(get_db)):
    posting, verdict = postings_service.capture_posting(
        db,
        company_name=payload.company_name,
        title=payload.title,
        raw_jd_text=payload.raw_jd_text,
        url=payload.url,
        posted_date=payload.posted_date,
        location=payload.location,
    )
    db.commit()
    db.refresh(posting)
    current_profile_version_id = profile_service.get_current_profile(db).id
    return PostingCreated(
        posting=_to_detail(posting, current_profile_version_id),
        verdict=build_verdict_out(verdict, current_profile_version_id),
    )


@router.get("/triage-pending/preview", response_model=BatchPreview)
def triage_pending_preview(
    company_id: int | None = None,
    posting_ids: list[int] | None = Query(None),
    limit: int | None = 30,
    db: Session = Depends(get_db),
):
    """Read-only: how many postings a POST /triage-pending call with these
    same params would actually submit, and an estimated cost -- computed
    without calling the Anthropic API or the relevance ranker (a preview
    only needs the count). Safe to call on every filter change."""
    return batch_preview_service.preview_stage2(
        db, company_id=company_id, posting_ids=posting_ids, limit=limit
    )


@router.post("/triage-pending", response_model=TriagePendingSubmitted)
def triage_pending(
    company_id: int | None = None,
    posting_ids: list[int] | None = Query(None),
    limit: int | None = 30,
    db: Session = Depends(get_db),
):
    """Submits the Stage-1-'analyze'/no-Stage-2 backlog as a single
    Anthropic Batch API request (50% discount vs. standard pricing) and
    returns immediately with the batch handle -- does not wait for
    results. Deliberately not run automatically during a crawl (see
    app/services/companies.py). Batches can take up to ~an hour; check back
    with POST /triage-pending/{batch_id}/apply whenever convenient. No
    automatic polling is built -- this is a manual, occasional operation.
    limit defaults to 30 (a safety cap, ranked by classical-NLP relevance,
    not cost) -- pass an explicit larger number to raise it. There's no
    HTTP-level way to fully disable it (query params can't express None);
    call stage2_batch.submit_pending_batch(limit=None) directly in Python
    if the pool genuinely needs to go uncapped."""
    try:
        result = stage2_batch.submit_pending_batch(
            db, company_id=company_id, posting_ids=posting_ids, limit=limit
        )
    except stage2_batch.Stage2UnavailableError:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured") from None
    db.commit()
    return result


@router.post("/triage-pending/{batch_id}/apply", response_model=TriagePendingApplied)
def apply_triage_batch(batch_id: str, db: Session = Depends(get_db)):
    """Checks the batch's status and, once ended, applies its results.
    Safe to call repeatedly before the batch ends -- returns the current
    status with nothing applied."""
    submission = batches_service.get_batch_submission(db, batch_id)
    apply_fn: Callable[..., dict]
    kind = submission.kind if submission is not None else None
    if kind == key_skills_backfill.KIND:
        apply_fn = key_skills_backfill.apply_backfill_results
    elif kind == key_skills_backfill.KIND_REFRESH:
        apply_fn = key_skills_backfill.apply_refresh_results
    else:
        apply_fn = stage2_batch.apply_batch_results
    try:
        result = apply_fn(db, batch_id)
    except stage2_batch.Stage2UnavailableError:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured") from None
    db.commit()
    return result


@router.post("/key-skills-backfill", response_model=TriagePendingSubmitted)
def key_skills_backfill_submit(
    posting_ids: list[int] | None = Query(None),
    db: Session = Depends(get_db),
):
    """Submits a Stage-2-style batch that records ONLY key_skills for
    postings that never got any (Stage-1 fast-tracked ones skip Stage 2) --
    see app/services/key_skills_backfill.py. Check/apply it through the
    same POST /triage-pending/{batch_id}/apply endpoint, which routes by
    the batch's recorded kind."""
    try:
        result = key_skills_backfill.submit_backfill_batch(db, posting_ids=posting_ids)
    except stage2_batch.Stage2UnavailableError:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured") from None
    db.commit()
    return result


@router.post("/key-skills-refresh", response_model=TriagePendingSubmitted)
def key_skills_refresh_submit(db: Session = Depends(get_db)):
    """Re-extracts key_skills with the current prompt for the interest set
    (manual captures, starred, actively applied to), overwriting the old
    ones. Check/apply through POST /triage-pending/{batch_id}/apply."""
    try:
        result = key_skills_backfill.submit_refresh_batch(db)
    except stage2_batch.Stage2UnavailableError:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured") from None
    db.commit()
    return result


@router.get("/analyze-pending/preview", response_model=BatchPreview)
def analyze_pending_preview(
    company_id: int | None = None,
    posting_ids: list[int] | None = Query(None),
    limit: int | None = 30,
    include_stale: bool = False,
    include_unready: bool = False,
    db: Session = Depends(get_db),
):
    """Read-only equivalent of /triage-pending/preview for Stage 3.
    include_stale (4.4): preview a deliberate re-analysis after a profile
    edit -- see submit_pending_batch's docstring. include_unready: also
    count postings Stage 1/2 did not pass (the default is "ready" only)."""
    return batch_preview_service.preview_stage3(
        db,
        company_id=company_id,
        posting_ids=posting_ids,
        limit=limit,
        include_stale=include_stale,
        include_unready=include_unready,
    )


@router.post("/analyze-pending", response_model=AnalyzePendingSubmitted)
def analyze_pending(
    company_id: int | None = None,
    posting_ids: list[int] | None = Query(None),
    limit: int | None = 30,
    include_stale: bool = False,
    include_unready: bool = False,
    db: Session = Depends(get_db),
):
    """Stage 3's batch equivalent of /triage-pending -- submits postings
    with no stage3_result yet as a single Anthropic Batch API request (50%
    discount) and returns immediately with the batch handle. Only "ready"
    postings (Stage 1 fast-track or a Stage 2 pass) unless
    include_unready=True. Check back with
    POST /analyze-pending/{batch_id}/apply whenever convenient; no
    automatic polling. limit defaults to 30 (relevance-ranked safety cap,
    same reasoning as /triage-pending) -- call
    stage3_batch.submit_pending_batch(limit=None) directly in Python to go
    uncapped. include_stale=True (4.4): also re-analyze postings whose
    Stage 3 result is stale against the current profile version -- never
    the default; a deliberate re-fit after a profile edit."""
    try:
        result = stage3_batch.submit_pending_batch(
            db,
            company_id=company_id,
            posting_ids=posting_ids,
            limit=limit,
            include_stale=include_stale,
            include_unready=include_unready,
        )
    except stage3_batch.Stage2UnavailableError:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured") from None
    db.commit()
    return result


@router.post("/analyze-pending/{batch_id}/apply", response_model=AnalyzePendingApplied)
def apply_analyze_batch(batch_id: str, db: Session = Depends(get_db)):
    """Checks the batch's status and, once ended, applies its results.
    Safe to call repeatedly before the batch ends -- returns the current
    status with nothing applied."""
    try:
        result = stage3_batch.apply_batch_results(db, batch_id)
    except stage3_batch.Stage2UnavailableError:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured") from None
    db.commit()
    return result


@router.post("/hydrate", response_model=HydrateResult)
def hydrate_postings(
    company_id: int | None = None,
    posting_ids: list[int] | None = Query(None),
    limit: int = 25,
    db: Session = Depends(get_db),
):
    """Fetches full detail pages for scraped postings that only have a
    short listing-page blurb, replacing raw_jd_text (+ locations, + posted
    date) and marking detail_fetched_at. Chunked via limit (default 25) --
    hydrating everything (Apple has ~2,000 unhydrated postings) at a
    polite ~1 req/s would hold this request open for tens of minutes, so
    the caller loops this, checking `remaining` each time, rather than one
    huge synchronous call. posting_ids narrows to specific postings (the
    desktop detail view's "Fetch full description" button targets just
    the one posting being viewed)."""
    result = postings_service.hydrate_scraped_postings(
        db, company_id=company_id, posting_ids=posting_ids, limit=limit
    )
    # Hydrated postings have real text now, worth reflecting in relevance --
    # cheap (measured ~0.85s over the whole real 2,744-posting corpus), so
    # just always do it rather than making the caller remember to.
    relevance_service.rescore_all(db)
    db.commit()
    return result


@router.post("/rescore", response_model=RescoreResult)
def rescore_postings(db: Session = Depends(get_db)):
    """Recomputes relevance_score for every posting against the current
    profile. Also triggered automatically after crawl/scrape/hydrate (new
    text) -- this is for triggering it manually after a profile edit, or
    just to force a refresh."""
    result = relevance_service.rescore_all(db)
    db.commit()
    return result


@router.get("/spend-summary", response_model=SpendSummary)
def spend_summary(db: Session = Depends(get_db)):
    """Tokens + estimated $ spent so far on Stage 2/Stage 3, all-time and
    last 30 days. See pricing.spend_summary for the estimation caveats."""
    return pricing_service.spend_summary(db)


@router.post("/{posting_id}/analyze", response_model=VerdictOut)
def analyze_posting(posting_id: int, db: Session = Depends(get_db)):
    """Stage 3 -- manually triggered per posting, never run automatically
    anywhere. No stage1/stage2 precondition; see run_stage3_analysis."""
    posting = postings_service.get_posting(db, posting_id)
    if posting is None:
        raise HTTPException(status_code=404, detail="Posting not found")
    if not posting.verdicts:
        raise HTTPException(status_code=409, detail="Posting has no verdict row yet")
    try:
        verdict = postings_service.run_stage3_analysis(db, posting, posting.verdicts[0])
    except stage3_service.Stage2UnavailableError:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured") from None
    db.commit()
    db.refresh(verdict)
    return build_verdict_out(verdict, profile_service.get_current_profile(db).id)


@router.get("/{posting_id}", response_model=PostingDetail)
def get_posting(posting_id: int, db: Session = Depends(get_db)):
    posting = postings_service.get_posting(db, posting_id)
    if posting is None:
        raise HTTPException(status_code=404, detail="Posting not found")
    return _to_detail(posting, profile_service.get_current_profile(db).id)


@router.get("/{posting_id}/similar", response_model=list[SimilarPosting])
def similar_postings(posting_id: int, limit: int = 10, db: Session = Depends(get_db)):
    """TF-IDF cosine similarity over normalized_jd_text + title across the
    whole corpus (2.5) -- purely informational, never a merge/dedup
    signal. See relevance.find_similar_postings and
    SIMILAR_ROLE_SAME_COMPANY_THRESHOLD for what possibly_same_role means
    and its known limitations."""
    result = relevance_service.find_similar_postings(db, posting_id, limit=limit)
    if result is None:
        raise HTTPException(status_code=404, detail="Posting not found")
    return [
        SimilarPosting(posting=_to_summary(p), similarity=sim, possibly_same_role=possibly_same)
        for p, sim, possibly_same in result
    ]


@router.get("", response_model=list[PostingSummary])
def list_postings(
    response: Response,
    company: str | None = None,
    starred: bool | None = None,
    url: str | None = None,
    posted_after: date | None = None,
    first_seen_after: date | None = None,
    keyword: str | None = None,
    keyword_scope: str = "title",
    location: str | None = None,
    workplace_type: str | None = None,
    location_match: bool | None = None,
    comp_min: int | None = None,
    has_comp: bool | None = None,
    open_only: bool = True,
    include_dismissed: bool = False,
    stage3_verdict: list[str] | None = Query(None),
    stage2_result: list[str] | None = Query(None),
    source: str | None = None,
    department: str | None = None,
    sort: str = "recent",
    limit: int | None = 200,
    db: Session = Depends(get_db),
):
    """first_seen_after is when this app first ingested the posting (a
    crawl/scrape/capture), posted_after is the employer's own posted date.
    keyword matches title + department (substring, case-insensitive);
    keyword_scope="all" also searches raw_jd_text. location_match=true
    uses the current profile's criteria.location_prefs (2.1) -- a posting
    with no location data at all still shows (neither matched nor
    hidden), only a definite non-match is excluded. open_only (default
    true) excludes closed_at postings; include_dismissed (default false)
    excludes dismissed_at ones. sort is one of recent|posted|fit|comp|
    relevance. X-Total-Count on the response reflects the filtered count
    before limit was applied, so a capped response is distinguishable
    from "that's really all of them"."""
    filters: dict[str, Any] = dict(
        company=company,
        starred=starred,
        url=url,
        posted_after=posted_after,
        first_seen_after=first_seen_after,
        keyword=keyword,
        keyword_scope=keyword_scope,
        location=location,
        workplace_type=workplace_type,
        location_match=location_match,
        comp_min=comp_min,
        has_comp=has_comp,
        open_only=open_only,
        include_dismissed=include_dismissed,
        stage3_verdict=stage3_verdict,
        stage2_result=stage2_result,
        source=source,
        department=department,
    )
    try:
        rows = postings_service.list_postings(db, sort=sort, limit=limit, **filters)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    total = postings_service.count_postings(db, **filters)
    response.headers["X-Total-Count"] = str(total)
    return [_to_summary(p) for p in rows]


@router.post("/{posting_id}/note", response_model=PostingSummary)
def set_posting_note(posting_id: int, note: str | None = None, db: Session = Depends(get_db)):
    posting = postings_service.set_note(db, posting_id, note)
    if posting is None:
        raise HTTPException(status_code=404, detail="Posting not found")
    db.commit()
    db.refresh(posting)
    return _to_summary(posting)


@router.post("/{posting_id}/star", response_model=PostingSummary)
def star_posting(posting_id: int, note: str | None = None, db: Session = Depends(get_db)):
    posting = postings_service.star_posting(db, posting_id, note=note)
    if posting is None:
        raise HTTPException(status_code=404, detail="Posting not found")
    db.commit()
    db.refresh(posting)
    return _to_summary(posting)


@router.post("/{posting_id}/unstar", response_model=PostingSummary)
def unstar_posting(posting_id: int, db: Session = Depends(get_db)):
    posting = postings_service.unstar_posting(db, posting_id)
    if posting is None:
        raise HTTPException(status_code=404, detail="Posting not found")
    db.commit()
    db.refresh(posting)
    return _to_summary(posting)


@router.post("/{posting_id}/dismiss", response_model=PostingSummary)
def dismiss_posting(posting_id: int, db: Session = Depends(get_db)):
    posting = postings_service.dismiss_posting(db, posting_id)
    if posting is None:
        raise HTTPException(status_code=404, detail="Posting not found")
    db.commit()
    db.refresh(posting)
    return _to_summary(posting)


@router.post("/{posting_id}/undismiss", response_model=PostingSummary)
def undismiss_posting(posting_id: int, db: Session = Depends(get_db)):
    posting = postings_service.undismiss_posting(db, posting_id)
    if posting is None:
        raise HTTPException(status_code=404, detail="Posting not found")
    db.commit()
    db.refresh(posting)
    return _to_summary(posting)
