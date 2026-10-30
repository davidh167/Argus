import logging
from datetime import date, datetime, timezone

from sqlalchemy import ColumnElement, Text, case, cast, false, func, or_, select
from sqlalchemy.orm import Session

from app.crawlers.fetch import FetchStrategy
from app.models.company import Company
from app.models.posting import Posting
from app.models.profile_version import ProfileVersion
from app.models.verdict import Verdict
from app.schemas.profile import LocationPrefs, ProfileSnapshotData
from app.services import batches as batches_service
from app.services import compensation
from app.services import detail_fetch as detail_fetch_service
from app.services import profile as profile_service
from app.services import roadmap as roadmap_service
from app.services import stage2 as stage2_service
from app.services import stage3 as stage3_service
from app.services.companies import get_or_create_company
from app.services.jd_sections import build_normalized_jd_text, extract_jd_sections
from app.services.stage1 import classify_title
from app.utils import linkedin

logger = logging.getLogger(__name__)


def latest_verdict_id_subquery():
    """Scalar subquery: the id of a posting's most recently created
    verdict, for correlated use in a JOIN against Posting -- e.g.
    `select(Posting).join(Verdict, Verdict.id == latest_verdict_id_subquery())`.
    Factored out here because list_stage2_pending, list_stage3_pending,
    and fit_tier_case (2.4) all need the exact same "this posting's
    current verdict" join independently."""
    return (
        select(Verdict.id)
        .where(Verdict.posting_id == Posting.id)
        .order_by(Verdict.created_at.desc())
        .limit(1)
        .correlate(Posting)
        .scalar_subquery()
    )


def fit_tier_case():
    """SQL CASE expression for a 0-6 fit-rank tier (sort=fit),
    evaluated against whatever Verdict row the caller has already joined
    as a posting's latest verdict (see latest_verdict_id_subquery) --
    this function doesn't do the join itself, it's a plain expression
    over Verdict's columns for the caller to select/order by.

    Deliberately checks every Stage 3 verdict value FIRST, before any
    Stage 2-only tier: CASE WHEN takes the first match, and a posting can
    have both a stage2_result AND a stage3_result (Stage 3 has no
    stage1/stage2 gate), so a real Stage 3 verdict must always win over a
    same-row Stage 2 result, not just when Stage 2 happens to be null.

    This is a starting point, not a claim (see the plan): unscored (tier
    2) sits above Weak Fit (tier 1) because unknown beats known-weak for
    discovery. A stale result (see build_verdict_out) keeps its tier here
    -- staleness is a badge, not a rank demotion.
    """
    stage3_verdict = Verdict.stage3_result["verdict"].astext
    return case(
        (stage3_verdict == "Strong Fit", 6),
        (stage3_verdict == "Moderate Fit", 5),
        (stage3_verdict == "Weak Fit", 1),
        (stage3_verdict == "Skip", 0),
        (Verdict.stage2_result == "fast_track", 4),
        (Verdict.stage2_result == "analyze", 3),
        (Verdict.stage2_result == "skip", 0),
        (Verdict.stage1_result == "skip", 0),
        else_=2,
    )


def location_match_condition(prefs: LocationPrefs):
    """SQL boolean expression equivalent to
    app.utils.location.matches_location(prefs, workplace_type=posting.
    workplace_type, locations=posting.locations) is True -- kept as a
    SEPARATE implementation (not a shared helper) because that function
    operates on already-fetched Python values for single-posting display,
    while this builds a WHERE clause so LIMIT/COUNT stay correct for
    ?location_match=true (the plan's "everything happens in SQL"). Must
    stay behaviorally identical; tests/test_postings_service.py cross-
    checks both against the same fixture data.

    prefs.remote_ok/hybrid_ok/onsite_ok are plain Python bools known at
    query-build time, so a disabled flag just omits that clause entirely
    rather than embedding a SQL boolean literal.

    exclude_cities/exclude_countries are ANDed against the final OR of
    positive clauses (a hard "no," unconditional, same as
    matches_location's Python behavior) rather than folded into each
    positive clause individually -- one exclusion check applies uniformly
    regardless of which positive clause would have matched.
    """
    locations_text = cast(Posting.locations, Text)
    clauses = []

    if prefs.remote_ok:
        remote_clause = Posting.workplace_type == "remote"
        if prefs.countries:
            remote_clause = remote_clause & or_(
                *(locations_text.ilike(f"%{c}%") for c in prefs.countries)
            )
        clauses.append(remote_clause)

    if prefs.cities:
        city_clause = or_(*(locations_text.ilike(f"%{c}%") for c in prefs.cities))
        # IS DISTINCT FROM, not !=, so a NULL (unknown) workplace_type is
        # correctly treated as "not hybrid"/"not onsite" -- matching
        # matches_location's Python behavior, where unknown workplace_type
        # never triggers hybrid/onsite gating.
        if not prefs.hybrid_ok:
            city_clause = city_clause & Posting.workplace_type.is_distinct_from("hybrid")
        if not prefs.onsite_ok:
            city_clause = city_clause & Posting.workplace_type.is_distinct_from("onsite")
        clauses.append(city_clause)

    if not clauses:
        return false()  # nothing configured (remote_ok False, cities empty) -- never matches

    condition = or_(*clauses)

    exclude_fields = (*prefs.exclude_cities, *prefs.exclude_countries)
    if exclude_fields:
        condition = condition & ~or_(*(locations_text.ilike(f"%{c}%") for c in exclude_fields))

    return condition


def _resolve_comp(
    raw_jd_text: str | None,
    comp_min: int | None,
    comp_max: int | None,
    comp_currency: str | None,
    comp_period: str | None,
) -> tuple[int | None, int | None, str | None, str | None, str | None, str | None]:
    """The one place comp resolution happens, used by both upsert_posting
    (crawled/scraped-listing/manually-captured postings) and
    hydrate_scraped_postings (detail hydration), so every source
    gets it the same way rather than each caller reimplementing text
    extraction. comp_min/comp_max already given (an ATS-structured field --
    currently only Lever's salaryRange) are trusted as-is
    (comp_source='ats_field'); otherwise raw_jd_text is run through
    compensation.extract_compensation (comp_source='jd_text').

    Returns (comp_min, comp_max, comp_currency, comp_period, comp_source,
    comp_raw).
    """
    if comp_min is not None and comp_max is not None:
        return comp_min, comp_max, comp_currency, comp_period, "ats_field", None
    if raw_jd_text:
        extracted = compensation.extract_compensation(raw_jd_text)
        if extracted:
            return (
                extracted.comp_min,
                extracted.comp_max,
                extracted.comp_currency,
                extracted.comp_period,
                "jd_text",
                extracted.comp_raw,
            )
    return None, None, None, None, None, None


def upsert_posting(
    db: Session,
    company: Company,
    *,
    title: str,
    url: str | None,
    external_id: str | None,
    raw_jd_text: str | None,
    posted_date: date | None,
    source: str,
    locations: list[str] | None = None,
    workplace_type: str | None = None,
    department: str | None = None,
    comp_min: int | None = None,
    comp_max: int | None = None,
    comp_currency: str | None = None,
    comp_period: str | None = None,
) -> tuple[Posting, bool, bool, bool]:
    """Match within the company by external_id if given (crawlers), else by
    url (manual capture). On match: refresh last_seen_at/text/title, and if
    the posting had been marked closed (see companies.crawl_company /
    scraper_catalog.scrape_company's closure marking), clear closed_at and
    bump repost_count -- that's what a "repost" actually is: a posting that
    fell off the board and came back, not merely being seen again while
    still open (which used to bump repost_count on every single re-crawl,
    regardless of whether anything changed -- see freshness_flag's history).
    Near-duplicate detection across sources is not implemented.

    comp_min/comp_max/comp_currency/comp_period, when given, are an
    ATS-structured field (currently only Lever's salaryRange) and are
    trusted as-is (comp_source='ats_field'). Otherwise, if raw_jd_text is
    present, compensation.extract_compensation runs against it
    (comp_source='jd_text') -- see _resolve_comp, used uniformly by every
    source (crawled, scraped via detail hydration, manually captured)
    rather than each reimplementing text extraction.

    Returns (posting, created, title_changed, reopened). title_changed and
    reopened are only meaningful when created is False. title_changed tells
    the caller whether the title differs from what was already stored,
    which matters because Stage 1 classification (classify_title) is
    title-only, so a re-crawl should only write a fresh verdict when the
    title actually changed (or the posting is brand new); otherwise it
    would just stack an identical verdict on every re-crawl. See
    write_stage1_verdict's callers.
    """
    comp_min, comp_max, comp_currency, comp_period, comp_source, comp_raw = _resolve_comp(
        raw_jd_text, comp_min, comp_max, comp_currency, comp_period
    )

    existing = None
    if external_id:
        existing = db.scalar(
            select(Posting).where(
                Posting.company_id == company.id, Posting.external_id == external_id
            )
        )
    elif url:
        existing = db.scalar(
            select(Posting).where(Posting.company_id == company.id, Posting.url == url)
        )

    if existing:
        title_changed = existing.title != title
        reopened = existing.closed_at is not None
        existing.last_seen_at = datetime.now(timezone.utc)
        if reopened:
            existing.closed_at = None
            existing.repost_count += 1
        existing.title = title
        if raw_jd_text:
            existing.raw_jd_text = raw_jd_text
            existing.normalized_jd_text = build_normalized_jd_text(raw_jd_text)
        if posted_date:
            existing.posted_date = posted_date
        if locations:
            existing.locations = locations
        if workplace_type:
            existing.workplace_type = workplace_type
        if department:
            existing.department = department
        if comp_source is not None:
            # Refreshed as a unit -- these five columns describe one
            # measurement, so a partial overwrite would mix an old
            # currency with a new amount.
            existing.comp_min = comp_min
            existing.comp_max = comp_max
            existing.comp_currency = comp_currency
            existing.comp_period = comp_period
            existing.comp_source = comp_source
            existing.comp_raw = comp_raw
        return existing, False, title_changed, reopened

    posting = Posting(
        company_id=company.id,
        title=title,
        url=url,
        external_id=external_id,
        raw_jd_text=raw_jd_text,
        normalized_jd_text=build_normalized_jd_text(raw_jd_text),
        posted_date=posted_date,
        source=source,
        locations=locations,
        workplace_type=workplace_type,
        department=department,
        comp_min=comp_min,
        comp_max=comp_max,
        comp_currency=comp_currency,
        comp_period=comp_period,
        comp_source=comp_source,
        comp_raw=comp_raw,
    )
    db.add(posting)
    db.flush()
    return posting, True, False, False


def mark_closed_postings(
    db: Session,
    company: Company,
    source: str,
    seen_external_ids: set[str],
    seen_urls: set[str],
) -> int:
    """Closes (sets closed_at) every currently-open posting of this
    company+source that wasn't seen in this crawl -- it fell off the board.
    Matches identity the same way upsert_posting does: external_id when the
    posting has one, else url.

    Only call this when the crawl is AUTHORITATIVE for this company+source
    -- a full, complete listing, not a partial one -- otherwise a posting
    that's still open but simply wasn't covered by a partial/capped/failed
    fetch would be wrongly marked closed. See crawl_company/scrape_company
    for what "authoritative" means for each source.
    """
    stmt = select(Posting).where(
        Posting.company_id == company.id,
        Posting.source == source,
        Posting.closed_at.is_(None),
    )
    now = datetime.now(timezone.utc)
    closed = 0
    for posting in db.scalars(stmt):
        seen = (
            posting.external_id in seen_external_ids
            if posting.external_id
            else posting.url in seen_urls
        )
        if not seen:
            posting.closed_at = now
            closed += 1
    return closed


def hydrate_scraped_postings(
    db: Session,
    company_id: int | None = None,
    posting_ids: list[int] | None = None,
    limit: int = 25,
    fetch_strategy: FetchStrategy | None = None,
) -> dict:
    """Fetches each posting's own detail page and replaces its short
    listing-page blurb with the full JD text (+ locations, + posted date
    when the page has them), then runs the same comp resolution every
    other source goes through and marks detail_fetched_at.

    Only ever touches source='scraped' postings with no detail_fetched_at
    yet -- crawled and manually-captured postings already have full text
    from ingestion. Closed postings are skipped too: no point spending a
    fetch on a role that's already gone.

    posting_ids narrows to specific postings (used by scrape_company to
    hydrate only the postings it just created, inline, rather than
    whichever postings happen to be oldest-unhydrated for the company --
    see SCRAPE_INLINE_HYDRATE_CAP). Without it, ordered oldest-first by id
    so a repeated chunked call (the desktop UI, or scripts/
    backfill_hydration.py) makes steady forward progress.

    Chunked via `limit` (default 25), not all-at-once: at a polite ~1
    req/s, Apple's ~2,000 unhydrated postings would take ~35 minutes,
    longer than this synchronous backend can hold one request open for.

    Per-posting failures (fetch error, or every extraction tier failing)
    are counted, not raised -- one bad posting shouldn't stop the batch.
    """
    stmt = (
        select(Posting)
        .where(
            Posting.source == "scraped",
            Posting.detail_fetched_at.is_(None),
            Posting.closed_at.is_(None),
        )
        .order_by(Posting.id)
        .limit(limit)
    )
    if company_id is not None:
        stmt = stmt.where(Posting.company_id == company_id)
    if posting_ids is not None:
        stmt = stmt.where(Posting.id.in_(posting_ids))
    postings = list(db.scalars(stmt))

    fetch_strategy = fetch_strategy or detail_fetch_service.SafeFetch()
    hydrated = failed = 0
    for posting in postings:
        if not posting.url:
            failed += 1
            continue
        try:
            extraction = detail_fetch_service.fetch_and_extract_detail(
                posting.url, fetch_strategy=fetch_strategy
            )
        except Exception:
            logger.exception("Detail hydration failed for posting_id=%s", posting.id)
            failed += 1
            continue
        if extraction is None:
            failed += 1
            continue

        posting.raw_jd_text = extraction.raw_jd_text
        posting.normalized_jd_text = build_normalized_jd_text(extraction.raw_jd_text)
        if extraction.posted_date:
            posting.posted_date = extraction.posted_date
        if extraction.locations:
            posting.locations = extraction.locations

        comp_min, comp_max, comp_currency, comp_period, comp_source, comp_raw = _resolve_comp(
            extraction.raw_jd_text, None, None, None, None
        )
        if comp_source is not None:
            posting.comp_min = comp_min
            posting.comp_max = comp_max
            posting.comp_currency = comp_currency
            posting.comp_period = comp_period
            posting.comp_source = comp_source
            posting.comp_raw = comp_raw

        posting.detail_fetched_at = datetime.now(timezone.utc)
        hydrated += 1

    remaining_stmt = (
        select(func.count())
        .select_from(Posting)
        .where(
            Posting.source == "scraped",
            Posting.detail_fetched_at.is_(None),
            Posting.closed_at.is_(None),
        )
    )
    if company_id is not None:
        remaining_stmt = remaining_stmt.where(Posting.company_id == company_id)
    if posting_ids is not None:
        remaining_stmt = remaining_stmt.where(Posting.id.in_(posting_ids))
    remaining = db.scalar(remaining_stmt)

    db.flush()
    return {"hydrated": hydrated, "failed": failed, "remaining": remaining}


def write_stage1_verdict(db: Session, posting: Posting) -> Verdict:
    profile_version = profile_service.get_current_profile(db)
    criteria = ProfileSnapshotData.model_validate(profile_version.snapshot).criteria
    stage1_result, stage1_reason = classify_title(posting.title, criteria)
    verdict = Verdict(
        posting_id=posting.id,
        profile_version_id=profile_version.id,
        stage1_result=stage1_result,
        stage1_reason=stage1_reason,
    )
    db.add(verdict)
    db.flush()
    return verdict


def run_stage2_triage(db: Session, posting: Posting, verdict: Verdict) -> Verdict:
    """Opt-in second step, called only where Stage 2 should run (currently
    just capture_posting -- crawl_company deliberately never calls this, see
    app/services/companies.py). Skips (leaves stage2 fields None) only for
    the one condition we actually have: no API key configured. Any other
    failure (bad response shape, auth error, network error) propagates --
    let it fail loudly rather than silently leaving stage2_result null with
    no trace of why."""
    if verdict.stage1_result != "analyze" or verdict.stage2_result is not None:
        return verdict  # idempotent guard

    profile_version = db.get_one(ProfileVersion, verdict.profile_version_id)
    snapshot = ProfileSnapshotData.model_validate(profile_version.snapshot)
    sections = extract_jd_sections(posting.raw_jd_text or "")

    try:
        result, reason, key_skills, tokens = stage2_service.triage(
            title=posting.title,
            sections=sections,
            criteria=snapshot.criteria,
            education=snapshot.credentials.education,
            skills_core=snapshot.skills.core,
            skills_developing=snapshot.skills.developing,
        )
    except stage2_service.Stage2UnavailableError:
        logger.info("Skipping Stage 2 for posting_id=%s: no API key configured", posting.id)
        return verdict

    verdict.stage2_result = result
    verdict.stage2_reason = reason
    verdict.stage2_key_skills = key_skills
    verdict.stage2_tokens_used = tokens
    verdict.stage2_profile_version_id = profile_version.id
    db.flush()
    return verdict


def run_stage3_analysis(db: Session, posting: Posting, verdict: Verdict) -> Verdict:
    """Manually triggered only -- never called from capture_posting,
    crawl_company, or any batch path. No gating on stage1_result/
    stage2_result: spec says "only runs on your explicit trigger" with no
    stage-gate language, so a fast-tracked posting (no Stage 2 result) can
    run Stage 3 directly. Always uses the CURRENT profile version, not
    verdict.profile_version_id (which reflects Stage 1's inputs and may now
    be stale -- that's what is_stale is for). No idempotency guard: a
    manual re-trigger is expected to give a fresh result, e.g. after a
    profile edit. Stage2UnavailableError propagates uncaught; the router
    translates it to a clear error."""
    profile_version = profile_service.get_current_profile(db)
    profile = ProfileSnapshotData.model_validate(profile_version.snapshot)
    sections = extract_jd_sections(posting.raw_jd_text or "")
    roadmap_items = roadmap_service.get_in_progress_skills(db)

    result, tokens = stage3_service.analyze(
        title=posting.title, sections=sections, profile=profile, roadmap_items=roadmap_items
    )

    verdict.stage3_triggered_at = datetime.now(timezone.utc)
    verdict.stage3_result = result.model_dump(mode="json")
    verdict.stage3_tokens_used = tokens
    verdict.stage3_profile_version_id = profile_version.id
    db.flush()
    return verdict


def capture_posting(
    db: Session,
    *,
    company_name: str,
    title: str,
    raw_jd_text: str,
    url: str | None = None,
    posted_date: date | None = None,
    location: str | None = None,
) -> tuple[Posting, Verdict]:
    workplace_type = None
    if linkedin.is_linkedin_job_url(url):
        # The extension sends the whole page's text; cut it down to the
        # actual description (see app/utils/linkedin.py). Only fills
        # location when the caller didn't supply one -- a value the user
        # typed in the popup wins over one parsed from the page.
        cleaned = linkedin.clean_capture(url, raw_jd_text)
        url, raw_jd_text, workplace_type = cleaned.url, cleaned.text, cleaned.workplace_type
        location = location or cleaned.location

    company = get_or_create_company(db, company_name, url)
    posting, _created, _title_changed, _reopened = upsert_posting(
        db,
        company,
        title=title,
        url=url,
        external_id=None,
        raw_jd_text=raw_jd_text,
        posted_date=posted_date,
        source="manual_capture",
        workplace_type=workplace_type,
        # A single free-text location, not parsed/geocoded -- the user
        # typed or the extension scraped whatever the page showed. Don't
        # LLM-extract this, per the plan: it's a manual capture, the user
        # is already looking at the posting and can type it directly.
        locations=[location] if location else None,
    )
    verdict = write_stage1_verdict(db, posting)
    run_stage2_triage(db, posting, verdict)
    return posting, verdict


def get_posting(db: Session, posting_id: int) -> Posting | None:
    return db.get(Posting, posting_id)


def _apply_posting_filters(
    stmt,
    db: Session,
    *,
    company: str | None,
    starred: bool | None,
    url: str | None,
    posted_after: date | None,
    first_seen_after: date | None,
    keyword: str | None,
    keyword_scope: str,
    location: str | None,
    workplace_type: str | None,
    location_match: bool | None,
    comp_min: int | None,
    has_comp: bool | None,
    open_only: bool,
    include_dismissed: bool,
    stage3_verdict: list[str] | None,
    stage2_result: list[str] | None,
    source: str | None,
    department: str | None,
):
    """Shared by list_postings/count_postings (the query differs, the
    WHERE clauses don't) -- both callers already JOIN Verdict (the latest
    one per posting, via latest_verdict_id_subquery) before calling this,
    so stage3_verdict/stage2_result can reference Verdict's columns
    directly.
    """
    if company:
        stmt = stmt.where(Company.name.ilike(f"%{company}%"))
    if starred is not None:
        stmt = stmt.where(Posting.starred == starred)
    if url is not None:
        stmt = stmt.where(Posting.url == url)
    if posted_after is not None:
        stmt = stmt.where(Posting.posted_date >= posted_after)
    if first_seen_after is not None:
        # When WE first ingested it (a crawl/scrape/capture), unlike
        # posted_after, which is the employer's own posted date.
        stmt = stmt.where(Posting.first_seen_at >= first_seen_after)
    if keyword:
        # Title + department always; JD text only for keyword_scope="all"
        # -- a full-text ILIKE scan is a heavier query than most users
        # want by default, so title/department (the fast, high-precision
        # case) is the default scope.
        keyword_conditions = [
            Posting.title.ilike(f"%{keyword}%"),
            Posting.department.ilike(f"%{keyword}%"),
        ]
        if keyword_scope == "all":
            keyword_conditions.append(Posting.raw_jd_text.ilike(f"%{keyword}%"))
        stmt = stmt.where(or_(*keyword_conditions))
    if location:
        stmt = stmt.where(cast(Posting.locations, Text).ilike(f"%{location}%"))
    if workplace_type:
        stmt = stmt.where(Posting.workplace_type == workplace_type)
    if location_match:
        profile_version = profile_service.get_current_profile(db)
        prefs = ProfileSnapshotData.model_validate(profile_version.snapshot).criteria.location_prefs
        # "Neither matched nor hidden" (2.1): a posting with no location
        # data at all stays visible under this toggle too, not just in
        # plain display -- otherwise "show postings matching my location"
        # would silently hide every posting whose location just isn't
        # known yet, which is a data gap, not a real non-match.
        unknown_location = Posting.workplace_type.is_(None) & Posting.locations.is_(None)
        stmt = stmt.where(or_(location_match_condition(prefs), unknown_location))
    if comp_min is not None:
        stmt = stmt.where(Posting.comp_max >= comp_min)
    if has_comp is not None:
        stmt = stmt.where(Posting.comp_min.is_not(None) if has_comp else Posting.comp_min.is_(None))
    if open_only:
        stmt = stmt.where(Posting.closed_at.is_(None))
    if not include_dismissed:
        stmt = stmt.where(Posting.dismissed_at.is_(None))
    if stage3_verdict:
        stmt = stmt.where(Verdict.stage3_result["verdict"].astext.in_(stage3_verdict))
    if stage2_result:
        stmt = stmt.where(Verdict.stage2_result.in_(stage2_result))
    if source:
        stmt = stmt.where(Posting.source == source)
    if department:
        stmt = stmt.where(Posting.department.ilike(f"%{department}%"))
    return stmt


_SORTS = {
    "recent": lambda: (Posting.last_seen_at.desc(),),
    "posted": lambda: (Posting.posted_date.desc().nulls_last(),),
    "fit": lambda: (fit_tier_case().desc(), Posting.relevance_score.desc().nulls_last()),
    "comp": lambda: (Posting.comp_max.desc().nulls_last(),),
    "relevance": lambda: (Posting.relevance_score.desc().nulls_last(),),
}


def _order_by(sort: str):
    try:
        return _SORTS[sort]()
    except KeyError:
        raise ValueError(f"Unknown sort={sort!r}; expected one of {sorted(_SORTS)}") from None


def list_postings(
    db: Session,
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
    stage3_verdict: list[str] | None = None,
    stage2_result: list[str] | None = None,
    source: str | None = None,
    department: str | None = None,
    sort: str = "recent",
    limit: int | None = 200,
) -> list[Posting]:
    stmt = _apply_posting_filters(
        select(Posting).join(Company).join(Verdict, Verdict.id == latest_verdict_id_subquery()),
        db,
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
    stmt = stmt.order_by(*_order_by(sort))
    if limit is not None:
        stmt = stmt.limit(limit)
    return list(db.scalars(stmt))


def count_postings(
    db: Session,
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
    stage3_verdict: list[str] | None = None,
    stage2_result: list[str] | None = None,
    source: str | None = None,
    department: str | None = None,
) -> int:
    """Same filters as list_postings, minus limit/sort -- lets a caller
    show "X of N" when the list itself is capped."""
    # No risk of row duplication from the Verdict join to account for: the
    # correlated subquery matches at most one verdict id per posting, so a
    # plain count is correct.
    stmt = _apply_posting_filters(
        select(func.count())
        .select_from(Posting)
        .join(Company)
        .join(Verdict, Verdict.id == latest_verdict_id_subquery()),
        db,
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
    return db.scalar(stmt)


def list_stage2_pending(
    db: Session,
    company_id: int | None = None,
    posting_ids: list[int] | None = None,
    exclude_in_flight: bool = True,
) -> list[Posting]:
    """Postings whose latest verdict is Stage 1 'analyze' with Stage 2 not
    yet run -- the backlog the batch triage endpoint works through.
    Dismissed postings are excluded unconditionally (marked not
    interested, so never worth a paid triage call), even when posting_ids
    is given explicitly. Postings already in a submitted-but-unapplied
    Stage 2 batch are excluded (exclude_in_flight) so they aren't paid for
    twice."""
    stmt = (
        select(Posting)
        .join(Verdict, Verdict.id == latest_verdict_id_subquery())
        .where(
            Verdict.stage1_result == "analyze",
            Verdict.stage2_result.is_(None),
            Posting.dismissed_at.is_(None),
        )
    )
    if exclude_in_flight:
        stmt = stmt.where(
            Posting.id.not_in(batches_service.in_flight_posting_ids(db, "stage2_triage"))
        )
    if company_id is not None:
        stmt = stmt.where(Posting.company_id == company_id)
    if posting_ids is not None:
        stmt = stmt.where(Posting.id.in_(posting_ids))
    return list(db.scalars(stmt))


def list_stage3_pending(
    db: Session,
    company_id: int | None = None,
    posting_ids: list[int] | None = None,
    include_stale: bool = False,
    include_unready: bool = False,
    exclude_in_flight: bool = True,
) -> list[Posting]:
    """Postings whose latest verdict has no stage3_result yet -- the backlog
    the Stage 3 batch endpoint works through. By default only "ready"
    postings: Stage 1 fast-tracked, or Stage 2 passed it (fast_track/
    analyze). Without that gate the pool is dominated by roles Stage 2
    already skipped (thousands, vs. a couple hundred ready), and a
    relevance-capped batch spends its slots on them -- observed live.
    include_unready=True lifts the gate for a deliberate manual run.
    Postings already in a submitted-but-unapplied Stage 3 batch are excluded
    (exclude_in_flight) so they aren't paid for twice.

    include_stale=True additionally includes postings whose latest verdict
    HAS a stage3_result but it's stale (computed against a profile version
    that isn't current anymore, per stage3_profile_version_id) -- for
    deliberate re-analysis after a profile edit. Never the default: a
    result with unknown provenance (stage3_profile_version_id is null --
    legacy rows, or an untracked batch_id) is never treated as stale, since
    that would be guessing.

    Dismissed postings are excluded unconditionally, same as
    list_stage2_pending, even when posting_ids is given explicitly."""
    condition: ColumnElement[bool] = Verdict.stage3_result.is_(None)
    if include_stale:
        current_profile_version_id = profile_service.get_current_profile(db).id
        condition = or_(
            condition,
            Verdict.stage3_profile_version_id.is_not(None)
            & (Verdict.stage3_profile_version_id != current_profile_version_id),
        )
    stmt = (
        select(Posting)
        .join(Verdict, Verdict.id == latest_verdict_id_subquery())
        .where(condition, Posting.dismissed_at.is_(None))
    )
    if not include_unready:
        stmt = stmt.where(
            or_(
                Verdict.stage1_result == "fast_track",
                Verdict.stage2_result.in_(("fast_track", "analyze")),
            )
        )
    if exclude_in_flight:
        stmt = stmt.where(
            Posting.id.not_in(batches_service.in_flight_posting_ids(db, "stage3_analysis"))
        )
    if company_id is not None:
        stmt = stmt.where(Posting.company_id == company_id)
    if posting_ids is not None:
        stmt = stmt.where(Posting.id.in_(posting_ids))
    return list(db.scalars(stmt))


def star_posting(db: Session, posting_id: int, note: str | None = None) -> Posting | None:
    posting = db.get(Posting, posting_id)
    if posting is None:
        return None
    posting.starred = True
    posting.starred_at = datetime.now(timezone.utc)
    if note is not None:
        posting.note = note
    db.flush()
    return posting


def set_note(db: Session, posting_id: int, note: str | None) -> Posting | None:
    """Standalone note edit, distinct from star_posting's optional note
    param -- that one only ever fires alongside starring; this lets the
    detail view's note field save on its own without also starring."""
    posting = db.get(Posting, posting_id)
    if posting is None:
        return None
    posting.note = note
    db.flush()
    return posting


def unstar_posting(db: Session, posting_id: int) -> Posting | None:
    posting = db.get(Posting, posting_id)
    if posting is None:
        return None
    posting.starred = False
    posting.starred_at = None  # keep it meaningful as "since when," not history
    db.flush()
    return posting


def dismiss_posting(db: Session, posting_id: int) -> Posting | None:
    posting = db.get(Posting, posting_id)
    if posting is None:
        return None
    posting.dismissed_at = datetime.now(timezone.utc)
    db.flush()
    return posting


def undismiss_posting(db: Session, posting_id: int) -> Posting | None:
    posting = db.get(Posting, posting_id)
    if posting is None:
        return None
    posting.dismissed_at = None
    db.flush()
    return posting
