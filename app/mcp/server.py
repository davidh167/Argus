"""MCP server wrapping the FastAPI backend's service layer directly (same
process family, own DB session per tool call -- no HTTP hop for a
single-machine tool).

Built against `mcp` v2.0.0, where the old `FastMCP` class from earlier SDK
versions was renamed to `MCPServer` (still `mcp.server.MCPServer`, same
`@mcp.tool()` decorator pattern, `mcp.run()` defaults to stdio transport).
"""

from mcp.server import MCPServer

from app.db import SessionLocal
from app.schemas.application import ApplicationOut
from app.schemas.posting import PostingDetail, PostingSummary
from app.schemas.profile import ProfileSnapshot
from app.schemas.roadmap import SkillRoadmapOut
from app.services import applications as applications_service
from app.services import postings as postings_service
from app.services import profile as profile_service
from app.services import relevance as relevance_service
from app.services import roadmap as roadmap_service
from app.services.profile import NoProfileError
from app.services.stage2 import Stage2UnavailableError
from app.utils.freshness import freshness_flag

mcp = MCPServer("argus")


def _to_summary(posting) -> dict:
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
        # repost_count was missing here entirely -- every MCP tool that
        # returns a posting summary (list_postings, star_posting,
        # capture_posting, get_posting) raised a Pydantic ValidationError
        # on every call, since PostingSummary.repost_count has no default.
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
    ).model_dump(mode="json")


def _to_detail(posting, db) -> dict:
    summary = _to_summary(posting)
    latest_verdict = None
    if posting.verdicts:
        from app.schemas.verdict import build_verdict_out

        current_profile_version_id = profile_service.get_current_profile(db).id
        latest_verdict = build_verdict_out(
            posting.verdicts[0], current_profile_version_id
        ).model_dump(mode="json")
    return PostingDetail(
        **summary,
        raw_jd_text=posting.raw_jd_text,
        posted_date=posting.posted_date,
        first_seen_at=posting.first_seen_at,
        last_seen_at=posting.last_seen_at,
        department=posting.department,
        comp_source=posting.comp_source,
        comp_raw=posting.comp_raw,
        detail_fetched_at=posting.detail_fetched_at,
        relevance_matched_terms=posting.relevance_matched_terms,
        latest_verdict=latest_verdict,
    ).model_dump(mode="json")


@mcp.tool()
def capture_posting(
    company_name: str,
    title: str,
    raw_jd_text: str,
    url: str | None = None,
    location: str | None = None,
) -> dict:
    """Capture a job posting the user is looking at: upserts the company,
    creates the posting, runs the free Stage 1 title-rules filter, and
    returns the created posting with its verdict. Use this when the user
    pastes in a JD they want tracked and discussed. location is free text
    (e.g. "San Francisco, CA"), not geocoded -- pass it along only if the
    user told you where the role is, never guessed."""
    with SessionLocal() as db:
        posting, _verdict = postings_service.capture_posting(
            db,
            company_name=company_name,
            title=title,
            raw_jd_text=raw_jd_text,
            url=url,
            location=location,
        )
        db.commit()
        db.refresh(posting)
        return _to_detail(posting, db)


@mcp.tool()
def get_posting(posting_id: int) -> dict:
    """Full JD text, company, freshness flag, and latest Stage 1 verdict for
    one posting by id. Use this to pull up a specific posting's full detail
    to discuss fit or draft a tailored resume against it."""
    with SessionLocal() as db:
        posting = postings_service.get_posting(db, posting_id)
        if posting is None:
            return {"error": f"No posting with id={posting_id}"}
        return _to_detail(posting, db)


@mcp.tool()
def similar_postings(posting_id: int, limit: int = 10) -> list[dict]:
    """Postings most similar to this one (TF-IDF cosine over JD text +
    title), across all companies -- purely informational, never a merge/
    dedup signal (different postings are always different roles, even at
    the same company). Each result includes similarity (0-1) and
    possibly_same_role (True only for a same-company pair above a
    calibrated threshold -- worth a second look, not a certainty; see
    relevance.SIMILAR_ROLE_SAME_COMPANY_THRESHOLD for real calibration
    data and known limitations, e.g. it currently runs noisier for
    companies whose JD text shares more cross-posting boilerplate). Use
    this to find other roles like one the user is already looking at."""
    with SessionLocal() as db:
        result = relevance_service.find_similar_postings(db, posting_id, limit=limit)
        if result is None:
            return [{"error": f"No posting with id={posting_id}"}]
        return [
            {**_to_summary(posting), "similarity": sim, "possibly_same_role": possibly_same}
            for posting, sim, possibly_same in result
        ]


@mcp.tool()
def list_postings(
    company: str | None = None,
    starred: bool | None = None,
    keyword: str | None = None,
    location: str | None = None,
    workplace_type: str | None = None,
    comp_min: int | None = None,
    has_comp: bool | None = None,
    stage3_verdict: list[str] | None = None,
    open_only: bool = True,
    include_dismissed: bool = False,
    sort: str = "recent",
    limit: int = 50,
) -> list[dict]:
    """List captured postings, filtered/sorted server-side (SQL, not a
    Python scan) so this stays cheap even against thousands of postings.
    workplace_type is 'onsite'|'hybrid'|'remote'; stage3_verdict is a list
    of any of 'Strong Fit'|'Moderate Fit'|'Weak Fit'|'Skip'; sort is
    'recent'|'posted'|'fit'|'comp'|'relevance'. open_only (default true)
    excludes roles that have closed; include_dismissed (default false)
    excludes postings the user marked not interested. Use this to see
    what's already been captured, e.g. narrowed to "remote roles paying
    at least $150k" or "postings ranked by fit," before asking for a
    specific posting's detail."""
    with SessionLocal() as db:
        rows = postings_service.list_postings(
            db,
            company=company,
            starred=starred,
            keyword=keyword,
            location=location,
            workplace_type=workplace_type,
            comp_min=comp_min,
            has_comp=has_comp,
            stage3_verdict=stage3_verdict,
            open_only=open_only,
            include_dismissed=include_dismissed,
            sort=sort,
            limit=limit,
        )
        return [_to_summary(p) for p in rows]


@mcp.tool()
def get_profile() -> dict:
    """The current (latest) profile snapshot: experience, projects, skills
    (core/developing/coursework), credentials, and fit criteria (skip
    patterns, target signals, seniority range). Use this before assessing fit
    on any posting -- it's the candidate's real, structured background."""
    with SessionLocal() as db:
        try:
            version = profile_service.get_current_profile(db)
        except NoProfileError as e:
            return {"error": str(e)}
        return ProfileSnapshot.model_validate(version).model_dump(mode="json")


@mcp.tool()
def set_posting_note(posting_id: int, note: str | None = None) -> dict:
    """Set or clear a posting's note without changing its starred status --
    use this for a plain annotation; use star_posting's own note param when
    the note accompanies marking interest."""
    with SessionLocal() as db:
        posting = postings_service.set_note(db, posting_id, note)
        if posting is None:
            return {"error": f"No posting with id={posting_id}"}
        db.commit()
        db.refresh(posting)
        return _to_summary(posting)


@mcp.tool()
def star_posting(posting_id: int, note: str | None = None) -> dict:
    """Mark a posting as an explicit interest signal (distinct from a fit
    verdict) -- the user's deliberate 'I want this kind of role,' with an
    optional note. Use this when the user says they're specifically
    interested in a posting you've discussed."""
    with SessionLocal() as db:
        posting = postings_service.star_posting(db, posting_id, note=note)
        if posting is None:
            return {"error": f"No posting with id={posting_id}"}
        db.commit()
        db.refresh(posting)
        return _to_summary(posting)


@mcp.tool()
def unstar_posting(posting_id: int) -> dict:
    """Clear a posting's starred interest signal. Use this when the user
    says a previously starred posting no longer interests them."""
    with SessionLocal() as db:
        posting = postings_service.unstar_posting(db, posting_id)
        if posting is None:
            return {"error": f"No posting with id={posting_id}"}
        db.commit()
        db.refresh(posting)
        return _to_summary(posting)


@mcp.tool()
def dismiss_posting(posting_id: int) -> dict:
    """Mark a posting as not of interest -- distinct from starring
    (dismissed postings are excluded from Stage 2/3 batch submission and
    skill-trend evidence). Use this when the user says they're not
    interested in a posting you've discussed."""
    with SessionLocal() as db:
        posting = postings_service.dismiss_posting(db, posting_id)
        if posting is None:
            return {"error": f"No posting with id={posting_id}"}
        db.commit()
        db.refresh(posting)
        return _to_summary(posting)


@mcp.tool()
def undismiss_posting(posting_id: int) -> dict:
    """Clear a posting's dismissed status. Use this when the user changes
    their mind about a posting they'd previously said isn't of interest."""
    with SessionLocal() as db:
        posting = postings_service.undismiss_posting(db, posting_id)
        if posting is None:
            return {"error": f"No posting with id={posting_id}"}
        db.commit()
        db.refresh(posting)
        return _to_summary(posting)


@mcp.tool()
def update_application_status(posting_id: int, status: str) -> dict:
    """Record or update the application-tracking status for a posting
    (e.g. saved/applied/screening/interview/rejected/skipped) -- one row
    per posting, upserted and mutated in place as status changes over time.
    Use this when the user tells you they've applied, heard back, gotten an
    interview, etc. for a posting already captured."""
    with SessionLocal() as db:
        if postings_service.get_posting(db, posting_id) is None:
            return {"error": f"No posting with id={posting_id}"}
        application = applications_service.update_application_status(db, posting_id, status)
        db.commit()
        db.refresh(application)
        return ApplicationOut.from_application(application).model_dump(mode="json")


@mcp.tool()
def detect_skill_trends(threshold: int = 3) -> list[dict]:
    """Skills appearing as a hard or soft gap in >= threshold starred
    postings' Stage 3 results, excluding anything already a core skill.
    Read-only, no LLM call. Use this to see what's worth pursuing before
    generating a direction brief for one."""
    with SessionLocal() as db:
        trends = roadmap_service.detect_skill_trends(db, threshold=threshold)
        return [t.model_dump(mode="json") for t in trends]


@mcp.tool()
def generate_roadmap_brief(
    skill_or_language: str, trigger_type: str, trend_evidence: list[int]
) -> dict:
    """Generate (or refresh) a direction brief for one skill roadmap item:
    what it is, why it matters, suggested scope/stack, and whether it
    strengthens an existing project. trigger_type is 'job_market_gap' or
    'project_strengthener'; trend_evidence is the starred posting ids that
    surfaced it (typically from detect_skill_trends). Use this after
    detect_skill_trends flags something worth pursuing."""
    with SessionLocal() as db:
        try:
            roadmap = roadmap_service.generate_roadmap_brief(
                db, skill_or_language, trigger_type=trigger_type, trend_evidence=trend_evidence
            )
        except Stage2UnavailableError as e:
            return {"error": str(e)}
        db.commit()
        db.refresh(roadmap)
        return SkillRoadmapOut.model_validate(roadmap).model_dump(mode="json")


@mcp.tool()
def update_roadmap_status(roadmap_id: int, status: str) -> dict:
    """Update a skill roadmap item's lifecycle status
    (flagged/researching/in_progress/complete). Marking an item
    'in_progress' makes it feed into future Stage 3 fit analyses as a
    weighted-but-caveated skill. Use this when the user says they've
    started or finished working on a roadmap skill."""
    with SessionLocal() as db:
        roadmap = roadmap_service.update_roadmap_status(db, roadmap_id, status)
        if roadmap is None:
            return {"error": f"No roadmap item with id={roadmap_id}"}
        db.commit()
        db.refresh(roadmap)
        return SkillRoadmapOut.model_validate(roadmap).model_dump(mode="json")
