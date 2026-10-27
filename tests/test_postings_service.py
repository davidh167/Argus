from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from app.crawlers.fetch import FetchStrategy
from app.db import engine
from app.schemas.stage3 import Stage3Result
from app.services.companies import get_or_create_company
from app.services.postings import (
    capture_posting,
    count_postings,
    dismiss_posting,
    fit_tier_case,
    hydrate_scraped_postings,
    latest_verdict_id_subquery,
    list_postings,
    list_stage2_pending,
    list_stage3_pending,
    location_match_condition,
    run_stage3_analysis,
    set_note,
    star_posting,
    undismiss_posting,
    unstar_posting,
    upsert_posting,
    write_stage1_verdict,
)
from app.services.profile import create_profile_version
from tests.conftest import TITLE_RULES

EMPTY_SNAPSHOT = {
    "experience": [],
    "projects": [],
    "skills": {"core": [], "developing": [], "coursework": []},
    "credentials": {"education": [], "oss_contributions": [], "publications": []},
    "criteria": {
        "skip_patterns": [],
        "target_signals": [],
        **TITLE_RULES,
        "location_prefs": {},
        "seniority_floor": None,
        "seniority_ceiling": None,
    },
}


@pytest.fixture
def db():
    """Each test runs inside its own transaction, rolled back at the end --
    exercises the real service layer against real Postgres without touching
    the dev DB's actual data."""
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    create_profile_version(session, snapshot=EMPTY_SNAPSHOT)  # verdicts need a profile_version_id
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def test_capture_posting_creates_company_and_verdict(db):
    posting, verdict = capture_posting(
        db,
        company_name="TestCo",
        title="ML Engineer",
        raw_jd_text="build models",
        url="https://testco.example.com/jobs/1",
    )
    assert posting.id is not None
    assert verdict.stage1_result == "fast_track"
    assert posting.company.name == "TestCo"
    # fast_track short-circuits Stage 2 entirely (see run_stage2_triage's guard)
    assert verdict.stage2_result is None


def test_capture_posting_runs_stage2_on_analyze_bucket(db):
    with patch("app.services.postings.stage2_service.triage") as mock_triage:
        mock_triage.return_value = ("skip", "canned reason for test", ["Python"], 42)
        posting, verdict = capture_posting(
            db,
            company_name="TestCo3",
            title="Software Engineer II",  # falls through Stage 1 to 'analyze'
            raw_jd_text="generic software engineering role",
            url="https://testco3.example.com/jobs/1",
        )
    assert verdict.stage1_result == "analyze"
    assert verdict.stage2_result == "skip"
    assert verdict.stage2_reason == "canned reason for test"
    assert verdict.stage2_key_skills == ["Python"]
    assert verdict.stage2_tokens_used == 42
    mock_triage.assert_called_once()


def test_capture_posting_passes_education_and_skills_to_stage2_triage(db):
    # Regression test: Stage 2 previously only sent seniority_floor/ceiling
    # (years of experience text) and target_signals (a short high-level
    # theme list), never the candidate's actual degree or real skill list --
    # so a JD's degree-based requirement ("MS + 2 years") or a specific
    # technology requirement could never be correctly judged against what
    # the candidate actually has.
    snapshot = {
        **EMPTY_SNAPSHOT,
        "credentials": {
            "education": ["MS Computer Science, Cal Poly SLO"],
            "oss_contributions": [],
            "publications": [],
        },
        "skills": {"core": ["PyTorch"], "developing": ["Kubernetes"], "coursework": []},
    }
    create_profile_version(db, snapshot=snapshot)
    with patch("app.services.postings.stage2_service.triage") as mock_triage:
        mock_triage.return_value = ("analyze", "ok", [], 10)
        capture_posting(
            db,
            company_name="TestCoEducation",
            title="Software Engineer II",
            raw_jd_text="generic software engineering role",
            url="https://testcoeducation.example.com/jobs/1",
        )
    assert mock_triage.call_args.kwargs["education"] == ["MS Computer Science, Cal Poly SLO"]
    assert mock_triage.call_args.kwargs["skills_core"] == ["PyTorch"]
    assert mock_triage.call_args.kwargs["skills_developing"] == ["Kubernetes"]


def test_capture_posting_records_stage2_profile_version_id(db):
    current = create_profile_version(db, snapshot=EMPTY_SNAPSHOT)
    with patch("app.services.postings.stage2_service.triage") as mock_triage:
        mock_triage.return_value = ("skip", "canned reason", [], 10)
        _posting, verdict = capture_posting(
            db,
            company_name="TestCo3b",
            title="Software Engineer II",
            raw_jd_text="generic software engineering role",
            url="https://testco3b.example.com/jobs/1",
        )
    # The Stage 1 verdict this Stage 2 result was written onto used
    # `current` as its profile_version_id (it's the latest at capture
    # time), so that's the value stage2_profile_version_id must record.
    assert verdict.profile_version_id == current.id
    assert verdict.stage2_profile_version_id == current.id


FAKE_STAGE3_RESULT = Stage3Result(
    verdict="Strong Fit",
    seniority_match="Yes",
    domain_match="Yes",
    stack_overlap=["Python"],
    hard_gaps=[],
    soft_gaps=[],
    developing_gaps=[],
    roadmap_gaps=[],
    roadmap_gap_caveat=None,
    fde_flag=False,
    resume_base="ml_search.tex",
    one_liner="Strong match.",
)


def test_run_stage3_analysis_writes_result_onto_latest_verdict(db):
    posting, verdict = capture_posting(
        db,
        company_name="TestCo4",
        title="ML Engineer",
        raw_jd_text="build models",
        url="https://testco4.example.com/jobs/1",
    )
    with patch("app.services.postings.stage3_service.analyze") as mock_analyze:
        mock_analyze.return_value = (FAKE_STAGE3_RESULT, 555)
        run_stage3_analysis(db, posting, verdict)

    assert verdict.stage3_triggered_at is not None
    assert verdict.stage3_result["verdict"] == "Strong Fit"
    assert verdict.stage3_tokens_used == 555


def test_run_stage3_analysis_records_profile_version_id(db):
    posting, verdict = capture_posting(
        db,
        company_name="TestCo4b",
        title="ML Engineer",
        raw_jd_text="build models",
        url="https://testco4b.example.com/jobs/1",
    )
    with patch("app.services.postings.stage3_service.analyze") as mock_analyze:
        mock_analyze.return_value = (FAKE_STAGE3_RESULT, 555)
        run_stage3_analysis(db, posting, verdict)

    from app.services.profile import get_current_profile

    assert verdict.stage3_profile_version_id == get_current_profile(db).id


def test_run_stage3_analysis_does_not_require_stage2_result(db):
    # fast_track short-circuits Stage 2 (see test_capture_posting_creates_company_and_verdict) --
    # Stage 3 must still run directly on it, proving there's no stage1/stage2 gate.
    posting, verdict = capture_posting(
        db,
        company_name="TestCo5",
        title="ML Engineer",
        raw_jd_text="build models",
        url="https://testco5.example.com/jobs/1",
    )
    assert verdict.stage1_result == "fast_track"
    assert verdict.stage2_result is None

    with patch("app.services.postings.stage3_service.analyze") as mock_analyze:
        mock_analyze.return_value = (FAKE_STAGE3_RESULT, 100)
        run_stage3_analysis(db, posting, verdict)  # must not raise

    assert verdict.stage3_result is not None


def test_list_stage3_pending_excludes_analyzed_postings_by_default(db):
    posting, verdict = capture_posting(
        db,
        company_name="TestCo9",
        title="ML Engineer",
        raw_jd_text="build models",
        url="https://testco9.example.com/jobs/1",
    )
    with patch("app.services.postings.stage3_service.analyze") as mock_analyze:
        mock_analyze.return_value = (FAKE_STAGE3_RESULT, 100)
        run_stage3_analysis(db, posting, verdict)

    pending = list_stage3_pending(db, posting_ids=[posting.id])
    assert posting.id not in [p.id for p in pending]


def test_list_stage3_pending_include_stale_surfaces_postings_from_an_old_profile_version(db):
    posting, verdict = capture_posting(
        db,
        company_name="TestCo10",
        title="ML Engineer",
        raw_jd_text="build models",
        url="https://testco10.example.com/jobs/1",
    )
    with patch("app.services.postings.stage3_service.analyze") as mock_analyze:
        mock_analyze.return_value = (FAKE_STAGE3_RESULT, 100)
        run_stage3_analysis(db, posting, verdict)

    stale_version_id = verdict.stage3_profile_version_id

    # Not stale yet -- still against the current profile version.
    assert list_stage3_pending(db, posting_ids=[posting.id], include_stale=True) == []

    create_profile_version(db, snapshot=EMPTY_SNAPSHOT)  # bumps "current"

    default_pending = list_stage3_pending(db, posting_ids=[posting.id])
    assert posting.id not in [p.id for p in default_pending]  # default never includes it

    stale_pending = list_stage3_pending(db, posting_ids=[posting.id], include_stale=True)
    assert posting.id in [p.id for p in stale_pending]
    assert verdict.stage3_profile_version_id == stale_version_id  # untouched by the bump


def test_list_stage3_pending_ignores_null_provenance_as_not_stale(db):
    # Legacy rows (or an untracked batch apply) leave stage3_profile_version_id
    # null -- unknown provenance must never be treated as stale, per this
    # codebase's no-speculative-fallbacks convention.
    posting, verdict = capture_posting(
        db,
        company_name="TestCo11",
        title="ML Engineer",
        raw_jd_text="build models",
        url="https://testco11.example.com/jobs/1",
    )
    with patch("app.services.postings.stage3_service.analyze") as mock_analyze:
        mock_analyze.return_value = (FAKE_STAGE3_RESULT, 100)
        run_stage3_analysis(db, posting, verdict)
    verdict.stage3_profile_version_id = None
    db.flush()

    create_profile_version(db, snapshot=EMPTY_SNAPSHOT)

    stale_pending = list_stage3_pending(db, posting_ids=[posting.id], include_stale=True)
    assert posting.id not in [p.id for p in stale_pending]


def test_upsert_posting_repost_count_unchanged_while_still_open(db):
    # repost_count means "reopened after being closed," not "seen again" --
    # a still-open posting re-upserted on every crawl must NOT bump it, or
    # every re-crawl inflates it regardless of whether anything closed.
    company = get_or_create_company(db, "TestCo2")

    p1, created1, title_changed1, reopened1 = upsert_posting(
        db,
        company,
        title="ML Engineer",
        url=None,
        external_id="ext-1",
        raw_jd_text="v1",
        posted_date=None,
        source="greenhouse_api",
    )
    assert created1 is True
    assert title_changed1 is False
    assert reopened1 is False
    assert p1.repost_count == 0

    p2, created2, title_changed2, reopened2 = upsert_posting(
        db,
        company,
        title="ML Engineer",
        url=None,
        external_id="ext-1",
        raw_jd_text="v2",
        posted_date=None,
        source="greenhouse_api",
    )
    assert created2 is False
    assert title_changed2 is False  # same title as before -- no verdict-worthy change
    assert reopened2 is False  # never closed -- just seen again
    assert p2.id == p1.id
    assert p2.repost_count == 0  # unchanged: still open the whole time
    assert p2.raw_jd_text == "v2"


def test_upsert_posting_reports_title_changed_on_repost(db):
    company = get_or_create_company(db, "TestCo7")

    p1, _, _, _ = upsert_posting(
        db,
        company,
        title="ML Engineer",
        url=None,
        external_id="ext-title-change",
        raw_jd_text="v1",
        posted_date=None,
        source="greenhouse_api",
    )

    p2, created2, title_changed2, reopened2 = upsert_posting(
        db,
        company,
        title="Senior ML Engineer",  # same external_id, different title
        url=None,
        external_id="ext-title-change",
        raw_jd_text="v1",
        posted_date=None,
        source="greenhouse_api",
    )
    assert created2 is False
    assert title_changed2 is True
    assert reopened2 is False
    assert p2.id == p1.id
    assert p2.title == "Senior ML Engineer"


def test_upsert_posting_bumps_repost_count_and_clears_closed_at_on_reopen(db):
    company = get_or_create_company(db, "TestCo12")

    p1, _, _, _ = upsert_posting(
        db,
        company,
        title="ML Engineer",
        url=None,
        external_id="ext-reopen",
        raw_jd_text="v1",
        posted_date=None,
        source="greenhouse_api",
    )
    assert p1.repost_count == 0

    from datetime import datetime, timezone

    p1.closed_at = datetime.now(timezone.utc)  # simulate a prior crawl closing it
    db.flush()

    p2, created2, title_changed2, reopened2 = upsert_posting(
        db,
        company,
        title="ML Engineer",
        url=None,
        external_id="ext-reopen",
        raw_jd_text="v2",
        posted_date=None,
        source="greenhouse_api",
    )
    assert created2 is False
    assert reopened2 is True
    assert p2.id == p1.id
    assert p2.closed_at is None
    assert p2.repost_count == 1


def test_upsert_posting_ats_field_comp_wins_over_jd_text(db):
    company = get_or_create_company(db, "AtsCompCo")
    # The JD text itself also contains a (different) dollar range -- the
    # explicit ats_field value must win, not the text extractor.
    posting, *_ = upsert_posting(
        db,
        company,
        title="ML Engineer",
        url=None,
        external_id="ats-comp-1",
        raw_jd_text="Unrelated text mentioning $10,000 - $20,000 somewhere.",
        posted_date=None,
        source="lever_api",
        comp_min=150_000,
        comp_max=180_000,
        comp_currency="USD",
        comp_period="year",
    )
    assert posting.comp_min == 150_000
    assert posting.comp_max == 180_000
    assert posting.comp_source == "ats_field"
    assert posting.comp_raw is None  # no matched snippet for a structured field


def test_upsert_posting_falls_back_to_jd_text_extraction(db):
    company = get_or_create_company(db, "JdTextCompCo")
    posting, *_ = upsert_posting(
        db,
        company,
        title="ML Engineer",
        url=None,
        external_id="jd-comp-1",
        raw_jd_text="Annual Salary:\n$270,000 — $340,000 USD",
        posted_date=None,
        source="greenhouse_api",
    )
    assert posting.comp_min == 270_000
    assert posting.comp_max == 340_000
    assert posting.comp_source == "jd_text"
    assert "270,000" in posting.comp_raw


def test_upsert_posting_no_comp_when_nothing_found(db):
    company = get_or_create_company(db, "NoCompCo")
    posting, *_ = upsert_posting(
        db,
        company,
        title="ML Engineer",
        url=None,
        external_id="no-comp-1",
        raw_jd_text="A generic job description with no salary mentioned.",
        posted_date=None,
        source="greenhouse_api",
    )
    assert posting.comp_min is None
    assert posting.comp_source is None


def test_upsert_posting_refreshes_location_workplace_department_when_present(db):
    company = get_or_create_company(db, "RefreshFieldsCo")
    p1, *_ = upsert_posting(
        db,
        company,
        title="ML Engineer",
        url=None,
        external_id="refresh-1",
        raw_jd_text="x",
        posted_date=None,
        source="greenhouse_api",
        locations=["San Francisco, CA"],
        workplace_type="onsite",
        department="Engineering",
    )
    assert p1.locations == ["San Francisco, CA"]

    # A re-crawl that omits these (e.g. a transient parsing gap) must not
    # blank out what's already stored.
    p2, *_ = upsert_posting(
        db,
        company,
        title="ML Engineer",
        url=None,
        external_id="refresh-1",
        raw_jd_text="x",
        posted_date=None,
        source="greenhouse_api",
    )
    assert p2.locations == ["San Francisco, CA"]
    assert p2.workplace_type == "onsite"
    assert p2.department == "Engineering"

    # A re-crawl WITH a new value does refresh it.
    p3, *_ = upsert_posting(
        db,
        company,
        title="ML Engineer",
        url=None,
        external_id="refresh-1",
        raw_jd_text="x",
        posted_date=None,
        source="greenhouse_api",
        locations=["Seattle, WA"],
        workplace_type="remote",
        department="Platform",
    )
    assert p3.locations == ["Seattle, WA"]
    assert p3.workplace_type == "remote"
    assert p3.department == "Platform"


def test_upsert_posting_leaves_existing_comp_untouched_when_nothing_new_derived(db):
    company = get_or_create_company(db, "CompPersistCo")
    p1, *_ = upsert_posting(
        db,
        company,
        title="ML Engineer",
        url=None,
        external_id="comp-persist-1",
        raw_jd_text="Annual Salary:\n$270,000 — $340,000 USD",
        posted_date=None,
        source="greenhouse_api",
    )
    assert p1.comp_min == 270_000

    # A re-crawl whose JD text this time has no parseable comp (e.g. a
    # scrape that only got a short blurb) must not erase the prior figure.
    p2, *_ = upsert_posting(
        db,
        company,
        title="ML Engineer",
        url=None,
        external_id="comp-persist-1",
        raw_jd_text="A generic blurb with no salary text.",
        posted_date=None,
        source="greenhouse_api",
    )
    assert p2.comp_min == 270_000
    assert p2.comp_max == 340_000


def test_upsert_posting_populates_normalized_jd_text(db):
    company = get_or_create_company(db, "NormalizedTextCo")
    raw = (
        "EEO BOILERPLATE ON EVERY POSTING. "
        "About the Role We build search systems. "
        "Minimum Qualifications 2+ years experience."
    )
    posting, *_ = upsert_posting(
        db,
        company,
        title="ML Engineer",
        url=None,
        external_id="normalized-1",
        raw_jd_text=raw,
        posted_date=None,
        source="greenhouse_api",
    )
    assert posting.normalized_jd_text is not None
    assert "search systems" in posting.normalized_jd_text
    assert "EEO BOILERPLATE" not in posting.normalized_jd_text

    # Refreshed in lockstep with raw_jd_text on a later upsert.
    posting2, *_ = upsert_posting(
        db,
        company,
        title="ML Engineer",
        url=None,
        external_id="normalized-1",
        raw_jd_text="About the Role Completely different text now. Minimum Qualifications 5+ years.",
        posted_date=None,
        source="greenhouse_api",
    )
    assert "Completely different text" in posting2.normalized_jd_text
    assert "search systems" not in posting2.normalized_jd_text


def test_capture_posting_stores_manual_location(db):
    posting, _verdict = capture_posting(
        db,
        company_name="ManualLocCo",
        title="ML Engineer",
        raw_jd_text="build models",
        url="https://manuallocco.example.com/1",
        location="Austin, TX",
    )
    assert posting.locations == ["Austin, TX"]


def test_capture_posting_no_location_leaves_locations_none(db):
    posting, _verdict = capture_posting(
        db,
        company_name="ManualNoLocCo",
        title="ML Engineer",
        raw_jd_text="build models",
        url="https://manualnolocco.example.com/1",
    )
    assert posting.locations is None


def test_capture_posting_no_location_stores_true_sql_null_not_json_null(db):
    """Regression test for a real bug found live: capture_posting passes
    locations=None explicitly to upsert_posting's Posting(...) constructor,
    and SQLAlchemy's JSON/JSONB type defaults to none_as_null=False -- an
    explicit Python None assigned to a JSON-typed column stores a literal
    JSON null value, not a true SQL NULL. `posting.locations is None`
    (the test above) can't catch this: it passes either way, since the
    in-memory Python attribute reads as None regardless. Only a query
    using .is_(None) (or checking jsonb_typeof directly) exposes it --
    which is exactly what broke: location_match_condition's "unknown
    location" detection (2.2) uses Posting.locations.is_(None), and
    `'null'::jsonb IS NULL` is false in Postgres, so an affected row read
    as a definite non-match instead of "no data." Fixed via
    none_as_null=True on every JSONB column (see Posting.locations)."""
    from sqlalchemy import select, text

    posting, _verdict = capture_posting(
        db,
        company_name="ManualNoLocSqlNullCo",
        title="ML Engineer",
        raw_jd_text="build models",
        url="https://manualnolocsqlnull.example.com/1",
    )
    typeof = db.execute(
        text("select jsonb_typeof(locations) from postings where id=:id"), {"id": posting.id}
    ).scalar()
    assert typeof is None  # NULL jsonb_typeof means the column itself is SQL NULL

    from app.models.posting import Posting

    assert (
        db.scalar(select(Posting.id).where(Posting.id == posting.id, Posting.locations.is_(None)))
        == posting.id
    )


class _MapFetch(FetchStrategy):
    """Serves canned HTML per exact URL, error text for anything unmapped."""

    def __init__(self, by_url: dict[str, str]):
        self.by_url = by_url
        self.calls: list[str] = []

    def fetch(self, url: str) -> str:
        self.calls.append(url)
        return self.by_url.get(url, "<html><body></body></html>")


_REAL_DETAIL_HTML = (
    "<html><body><h1>Full Detail</h1>"
    "<p>About the Role: This is the complete job description fetched from "
    "the posting's own detail page, much longer and richer than the short "
    "listing-page blurb it started with, describing the team and role.</p>"
    "</body></html>"
)


def _make_scraped_posting(db, company_name: str, url: str, blurb: str = "short blurb"):
    from app.services.companies import get_or_create_company as _goc

    company = _goc(db, company_name)
    posting, *_ = upsert_posting(
        db,
        company,
        title="ML Engineer",
        url=url,
        external_id=None,
        raw_jd_text=blurb,
        posted_date=None,
        source="scraped",
    )
    return posting


def test_hydrate_scraped_postings_replaces_blurb_with_full_text(db):
    posting = _make_scraped_posting(db, "HydrateCo1", "https://hydrateco1.example.com/jobs/1")
    fetch = _MapFetch({posting.url: _REAL_DETAIL_HTML})

    result = hydrate_scraped_postings(db, posting_ids=[posting.id], fetch_strategy=fetch)

    assert result["hydrated"] == 1
    assert result["failed"] == 0
    assert "complete job description" in posting.raw_jd_text.lower()
    assert posting.normalized_jd_text is not None
    assert posting.detail_fetched_at is not None


def test_hydrate_scraped_postings_only_touches_unhydrated_scraped_postings(db):
    company = get_or_create_company(db, "HydrateScopeCo")
    scraped, *_ = upsert_posting(
        db,
        company,
        title="A",
        url="https://hydratescope.example.com/1",
        external_id=None,
        raw_jd_text="blurb",
        posted_date=None,
        source="scraped",
    )
    crawled, *_ = upsert_posting(
        db,
        company,
        title="B",
        url=None,
        external_id="ext-crawled",
        raw_jd_text="already full text from the crawler",
        posted_date=None,
        source="greenhouse_api",
    )
    fetch = _MapFetch({scraped.url: _REAL_DETAIL_HTML})

    result = hydrate_scraped_postings(db, company_id=company.id, fetch_strategy=fetch)

    assert result["hydrated"] == 1
    assert scraped.detail_fetched_at is not None
    assert crawled.detail_fetched_at is None  # never touched -- wrong source
    assert fetch.calls == [scraped.url]  # crawled posting's (nonexistent) url never fetched


def test_hydrate_scraped_postings_skips_closed_postings(db):
    posting = _make_scraped_posting(db, "HydrateClosedCo", "https://hydrateclosed.example.com/1")
    from datetime import datetime, timezone

    posting.closed_at = datetime.now(timezone.utc)
    db.flush()
    fetch = _MapFetch({posting.url: _REAL_DETAIL_HTML})

    result = hydrate_scraped_postings(db, posting_ids=[posting.id], fetch_strategy=fetch)

    assert result["hydrated"] == 0
    assert fetch.calls == []  # never even fetched


def test_hydrate_scraped_postings_counts_extraction_failure_without_raising(db):
    posting = _make_scraped_posting(db, "HydrateFailCo", "https://hydratefail.example.com/1")
    fetch = _MapFetch({posting.url: "<html><body></body></html>"})  # every tier fails on this

    result = hydrate_scraped_postings(db, posting_ids=[posting.id], fetch_strategy=fetch)

    assert result["hydrated"] == 0
    assert result["failed"] == 1
    assert posting.detail_fetched_at is None
    assert posting.raw_jd_text == "short blurb"  # untouched


def test_hydrate_scraped_postings_counts_fetch_exception_without_raising(db):
    posting = _make_scraped_posting(db, "HydrateExcCo", "https://hydrateexc.example.com/1")

    class RaisingFetch(FetchStrategy):
        def fetch(self, url: str) -> str:
            raise ConnectionError("simulated network failure")

    result = hydrate_scraped_postings(db, posting_ids=[posting.id], fetch_strategy=RaisingFetch())

    assert result["hydrated"] == 0
    assert result["failed"] == 1
    assert posting.detail_fetched_at is None


def test_hydrate_scraped_postings_skips_posting_with_no_url(db):
    company = get_or_create_company(db, "HydrateNoUrlCo")
    posting, *_ = upsert_posting(
        db,
        company,
        title="A",
        url=None,
        external_id=None,
        raw_jd_text="blurb",
        posted_date=None,
        source="scraped",
    )
    result = hydrate_scraped_postings(db, posting_ids=[posting.id], fetch_strategy=_MapFetch({}))
    assert result["hydrated"] == 0
    assert result["failed"] == 1


def test_hydrate_scraped_postings_respects_limit_and_reports_remaining(db):
    company = get_or_create_company(db, "HydrateLimitCo")
    postings = []
    for i in range(4):
        p, *_ = upsert_posting(
            db,
            company,
            title=f"Role {i}",
            url=f"https://hydratelimit.example.com/{i}",
            external_id=None,
            raw_jd_text="blurb",
            posted_date=None,
            source="scraped",
        )
        postings.append(p)
    fetch = _MapFetch({p.url: _REAL_DETAIL_HTML for p in postings})

    result = hydrate_scraped_postings(db, company_id=company.id, limit=2, fetch_strategy=fetch)

    assert result["hydrated"] == 2
    assert result["remaining"] == 2


def test_hydrate_scraped_postings_sets_comp_and_locations_from_extraction(db):
    posting = _make_scraped_posting(db, "HydrateCompCo", "https://hydratecomp.example.com/1")
    html = (
        "<html><body><h1>Role</h1>"
        "<p>About the Role: Build things for a large distributed platform team "
        "working on infrastructure that serves millions of requests per day, "
        "collaborating closely with product, design, and other engineering "
        "teams across the organization to ship reliable, well-tested systems.</p>"
        "<p>Annual Salary: $150,000 - $180,000</p>"
        "</body></html>"
    )
    fetch = _MapFetch({posting.url: html})

    hydrate_scraped_postings(db, posting_ids=[posting.id], fetch_strategy=fetch)

    assert posting.comp_min == 150_000
    assert posting.comp_max == 180_000
    assert posting.comp_source == "jd_text"


def _posting_with_verdict(db, name, url, *, stage1=None, stage2=None, stage3_verdict=None):
    from app.models.verdict import Verdict
    from app.services.profile import get_current_profile

    company = get_or_create_company(db, name)
    posting, *_ = upsert_posting(
        db,
        company,
        title="ML Engineer",
        url=url,
        external_id=None,
        raw_jd_text="x",
        posted_date=None,
        source="greenhouse_api",
    )
    verdict = Verdict(
        posting_id=posting.id,
        profile_version_id=get_current_profile(db).id,
        stage1_result=stage1,
        stage2_result=stage2,
        stage3_result={"verdict": stage3_verdict} if stage3_verdict else None,
    )
    db.add(verdict)
    db.flush()
    return posting


def _fit_tier(db, posting_id) -> int:
    from sqlalchemy import select

    from app.models.posting import Posting
    from app.models.verdict import Verdict

    stmt = (
        select(fit_tier_case())
        .select_from(Posting)
        .join(Verdict, Verdict.id == latest_verdict_id_subquery())
        .where(Posting.id == posting_id)
    )
    return db.scalar(stmt)


def test_fit_tier_stage3_strong_fit_is_6(db):
    p = _posting_with_verdict(
        db, "FitTier6Co", "https://fittier6.example.com/1", stage3_verdict="Strong Fit"
    )
    assert _fit_tier(db, p.id) == 6


def test_fit_tier_stage3_moderate_fit_is_5(db):
    p = _posting_with_verdict(
        db, "FitTier5Co", "https://fittier5.example.com/1", stage3_verdict="Moderate Fit"
    )
    assert _fit_tier(db, p.id) == 5


def test_fit_tier_stage2_fast_track_no_stage3_is_4(db):
    p = _posting_with_verdict(
        db, "FitTier4Co", "https://fittier4.example.com/1", stage2="fast_track"
    )
    assert _fit_tier(db, p.id) == 4


def test_fit_tier_stage2_analyze_no_stage3_is_3(db):
    p = _posting_with_verdict(db, "FitTier3Co", "https://fittier3.example.com/1", stage2="analyze")
    assert _fit_tier(db, p.id) == 3


def test_fit_tier_unscored_stage1_only_is_2(db):
    p = _posting_with_verdict(db, "FitTier2Co", "https://fittier2.example.com/1", stage1="analyze")
    assert _fit_tier(db, p.id) == 2


def test_fit_tier_stage3_weak_fit_is_1(db):
    p = _posting_with_verdict(
        db, "FitTier1Co", "https://fittier1.example.com/1", stage3_verdict="Weak Fit"
    )
    assert _fit_tier(db, p.id) == 1


def test_fit_tier_stage3_skip_is_0(db):
    p = _posting_with_verdict(
        db, "FitTier0aCo", "https://fittier0a.example.com/1", stage3_verdict="Skip"
    )
    assert _fit_tier(db, p.id) == 0


def test_fit_tier_stage2_skip_is_0(db):
    p = _posting_with_verdict(db, "FitTier0bCo", "https://fittier0b.example.com/1", stage2="skip")
    assert _fit_tier(db, p.id) == 0


def test_fit_tier_stage1_skip_is_0(db):
    p = _posting_with_verdict(db, "FitTier0cCo", "https://fittier0c.example.com/1", stage1="skip")
    assert _fit_tier(db, p.id) == 0


def test_fit_tier_stage3_verdict_wins_over_stage2_result_regardless_of_order(db):
    # A posting CAN have both a real stage2_result and a stage3_result
    # (Stage 3 has no stage1/stage2 gate) -- Stage 3's verdict must win,
    # not just when stage2_result happens to be null.
    p = _posting_with_verdict(
        db,
        "FitTierBothCo",
        "https://fittierboth.example.com/1",
        stage1="fast_track",
        stage2="fast_track",
        stage3_verdict="Weak Fit",
    )
    assert _fit_tier(db, p.id) == 1  # Weak Fit, not 4 (fast_track)


# -- 2.2: filters and sort on GET /postings (list_postings/count_postings) --


def _posting_with_location(db, name, url, *, workplace_type=None, locations=None, comp_max=None):
    company = get_or_create_company(db, name)
    posting, *_ = upsert_posting(
        db,
        company,
        title="ML Engineer",
        url=url,
        external_id=None,
        raw_jd_text="x",
        posted_date=None,
        source="greenhouse_api",
        workplace_type=workplace_type,
        locations=locations,
    )
    write_stage1_verdict(db, posting)
    if comp_max is not None:
        posting.comp_min = comp_max
        posting.comp_max = comp_max
        posting.comp_source = "ats_field"
        db.flush()
    return posting


def test_location_match_condition_matches_python_matches_location_function(db):
    from app.schemas.profile import LocationPrefs
    from app.utils.location import matches_location

    scenarios = [
        # (workplace_type, locations, prefs)
        ("remote", ["United States"], LocationPrefs(remote_ok=True)),
        ("remote", ["United States"], LocationPrefs(remote_ok=False)),
        ("remote", ["United States"], LocationPrefs(remote_ok=True, countries=["Canada"])),
        ("remote", ["Remote, Canada"], LocationPrefs(remote_ok=True, countries=["Canada"])),
        ("onsite", ["San Francisco, CA"], LocationPrefs(cities=["San Francisco"], onsite_ok=True)),
        ("onsite", ["San Francisco, CA"], LocationPrefs(cities=["San Francisco"], onsite_ok=False)),
        ("hybrid", ["Seattle, WA"], LocationPrefs(cities=["Seattle"], hybrid_ok=False)),
        (None, ["Austin, TX"], LocationPrefs(cities=["Austin"], hybrid_ok=False, onsite_ok=False)),
        ("onsite", ["Denver, CO"], LocationPrefs(cities=["Seattle"])),
        (
            "onsite",
            ["New York, NY"],
            LocationPrefs(cities=["New York"], exclude_cities=["New York"]),
        ),
        (
            "remote",
            ["Remote, United States"],
            LocationPrefs(
                remote_ok=True, countries=["United States"], exclude_countries=["United States"]
            ),
        ),
        ("hybrid", ["Seattle, WA"], LocationPrefs(cities=["Seattle"], exclude_cities=["Portland"])),
    ]
    for i, (wt, locs, prefs) in enumerate(scenarios):
        p = _posting_with_location(
            db,
            f"LocMatchCo{i}",
            f"https://locmatchco{i}.example.com/1",
            workplace_type=wt,
            locations=locs,
        )
        expected = matches_location(prefs, workplace_type=wt, locations=locs)
        from sqlalchemy import select

        from app.models.posting import Posting

        sql_result = db.scalar(select(location_match_condition(prefs)).where(Posting.id == p.id))
        assert bool(sql_result) == bool(expected), (
            f"scenario {i}: wt={wt} locs={locs} prefs={prefs}"
        )


def test_list_postings_open_only_defaults_true(db):
    from datetime import datetime, timezone

    open_p = _posting_with_location(db, "OpenOnlyOpenCo", "https://openonlyopen.example.com/1")
    closed_p = _posting_with_location(
        db, "OpenOnlyClosedCo", "https://openonlyclosed.example.com/1"
    )
    closed_p.closed_at = datetime.now(timezone.utc)
    db.flush()

    ids_default = {p.id for p in list_postings(db, company="OpenOnly")}
    assert open_p.id in ids_default
    assert closed_p.id not in ids_default

    ids_all = {p.id for p in list_postings(db, company="OpenOnly", open_only=False)}
    assert open_p.id in ids_all
    assert closed_p.id in ids_all


def test_list_postings_first_seen_after_scopes_to_recently_ingested(db):
    from datetime import date, datetime, timedelta, timezone

    old = _posting_with_location(db, "FirstSeenOldCo", "https://firstseenold.example.com/1")
    old.first_seen_at = datetime.now(timezone.utc) - timedelta(days=30)
    new = _posting_with_location(db, "FirstSeenNewCo", "https://firstseennew.example.com/1")
    db.flush()

    cutoff = date.today() - timedelta(days=2)
    ids = {p.id for p in list_postings(db, company="FirstSeen", first_seen_after=cutoff)}
    assert new.id in ids and old.id not in ids
    assert count_postings(db, company="FirstSeen", first_seen_after=cutoff) == 1
    assert {p.id for p in list_postings(db, company="FirstSeen")} >= {new.id, old.id}


def test_list_postings_include_dismissed_defaults_false(db):
    visible = _posting_with_location(db, "InclDismVisCo", "https://incldismvis.example.com/1")
    hidden = _posting_with_location(db, "InclDismHidCo", "https://incldismhid.example.com/1")
    dismiss_posting(db, hidden.id)

    ids_default = {p.id for p in list_postings(db, company="InclDism")}
    assert visible.id in ids_default
    assert hidden.id not in ids_default

    ids_all = {p.id for p in list_postings(db, company="InclDism", include_dismissed=True)}
    assert visible.id in ids_all
    assert hidden.id in ids_all


def test_list_postings_filters_by_location_substring(db):
    sf = _posting_with_location(
        db, "LocFilterSFCo", "https://locfiltersf.example.com/1", locations=["San Francisco, CA"]
    )
    ny = _posting_with_location(
        db, "LocFilterNYCo", "https://locfilterny.example.com/1", locations=["New York, NY"]
    )
    matches = list_postings(db, location="San Francisco")
    match_ids = {p.id for p in matches}
    assert sf.id in match_ids
    assert ny.id not in match_ids


def test_list_postings_filters_by_workplace_type(db):
    remote = _posting_with_location(
        db, "WtFilterRemoteCo", "https://wtfilterremote.example.com/1", workplace_type="remote"
    )
    onsite = _posting_with_location(
        db, "WtFilterOnsiteCo", "https://wtfilteronsite.example.com/1", workplace_type="onsite"
    )
    matches = {p.id for p in list_postings(db, workplace_type="remote")}
    assert remote.id in matches
    assert onsite.id not in matches


def test_list_postings_comp_min_filters_by_comp_max_at_least(db):
    high = _posting_with_location(
        db, "CompMinHighCo", "https://compminhigh.example.com/1", comp_max=200_000
    )
    low = _posting_with_location(
        db, "CompMinLowCo", "https://compminlow.example.com/1", comp_max=50_000
    )
    matches = {p.id for p in list_postings(db, comp_min=100_000)}
    assert high.id in matches
    assert low.id not in matches


def test_list_postings_has_comp_filter(db):
    with_comp = _posting_with_location(
        db, "HasCompYesCo", "https://hascompyes.example.com/1", comp_max=150_000
    )
    without_comp = _posting_with_location(db, "HasCompNoCo", "https://hascompno.example.com/1")
    matches_true = {p.id for p in list_postings(db, has_comp=True)}
    matches_false = {p.id for p in list_postings(db, has_comp=False)}
    assert with_comp.id in matches_true and with_comp.id not in matches_false
    assert without_comp.id in matches_false and without_comp.id not in matches_true


def test_list_postings_filters_by_stage3_verdict(db):

    strong = _posting_with_location(db, "S3FilterStrongCo", "https://s3filterstrong.example.com/1")
    weak = _posting_with_location(db, "S3FilterWeakCo", "https://s3filterweak.example.com/1")
    strong.verdicts[0].stage3_result = {"verdict": "Strong Fit"}
    weak.verdicts[0].stage3_result = {"verdict": "Weak Fit"}
    db.flush()

    matches = {p.id for p in list_postings(db, stage3_verdict=["Strong Fit"])}
    assert strong.id in matches
    assert weak.id not in matches

    matches_multi = {p.id for p in list_postings(db, stage3_verdict=["Strong Fit", "Weak Fit"])}
    assert strong.id in matches_multi and weak.id in matches_multi


def test_list_postings_filters_by_stage2_result(db):
    fast = _posting_with_location(db, "S2FilterFastCo", "https://s2filterfast.example.com/1")
    skip = _posting_with_location(db, "S2FilterSkipCo", "https://s2filterskip.example.com/1")
    fast.verdicts[0].stage2_result = "fast_track"
    skip.verdicts[0].stage2_result = "skip"
    db.flush()

    matches = {p.id for p in list_postings(db, stage2_result=["fast_track"])}
    assert fast.id in matches
    assert skip.id not in matches


def test_list_postings_filters_by_source(db):
    manual, _ = capture_posting(
        db,
        company_name="SourceFilterManualCo",
        title="ML Engineer",
        raw_jd_text="x",
        url="https://sourcefiltermanual.example.com/1",
    )
    scraped = _posting_with_location(
        db, "SourceFilterScrapedCo", "https://sourcefilterscraped.example.com/1"
    )

    matches = {p.id for p in list_postings(db, source="manual_capture")}
    assert manual.id in matches
    assert scraped.id not in matches


def test_list_postings_filters_by_department(db):
    company = get_or_create_company(db, "DeptFilterCo")
    eng, *_ = upsert_posting(
        db,
        company,
        title="ML Engineer",
        url="https://deptfiltereng.example.com/1",
        external_id=None,
        raw_jd_text="x",
        posted_date=None,
        source="greenhouse_api",
        department="Engineering",
    )
    write_stage1_verdict(db, eng)
    sales, *_ = upsert_posting(
        db,
        company,
        title="AE",
        url="https://deptfiltersales.example.com/1",
        external_id=None,
        raw_jd_text="x",
        posted_date=None,
        source="greenhouse_api",
        department="Sales",
    )
    write_stage1_verdict(db, sales)

    matches = {p.id for p in list_postings(db, department="Engineer")}
    assert eng.id in matches
    assert sales.id not in matches


def test_list_postings_keyword_matches_title_and_department_by_default(db):
    company = get_or_create_company(db, "KwScopeCo")
    title_match, *_ = upsert_posting(
        db,
        company,
        title="Distributed Systems Engineer",
        url="https://kwscopetitle.example.com/1",
        external_id=None,
        raw_jd_text="unrelated body text",
        posted_date=None,
        source="greenhouse_api",
    )
    write_stage1_verdict(db, title_match)
    dept_match, *_ = upsert_posting(
        db,
        company,
        title="Generic Role",
        url="https://kwscopedept.example.com/1",
        external_id=None,
        raw_jd_text="unrelated body text",
        posted_date=None,
        source="greenhouse_api",
        department="Distributed Systems",
    )
    write_stage1_verdict(db, dept_match)
    body_only, *_ = upsert_posting(
        db,
        company,
        title="Another Role",
        url="https://kwscopebody.example.com/1",
        external_id=None,
        raw_jd_text="mentions distributed systems deep in the body",
        posted_date=None,
        source="greenhouse_api",
    )
    write_stage1_verdict(db, body_only)

    default_matches = {p.id for p in list_postings(db, keyword="distributed systems")}
    assert title_match.id in default_matches
    assert dept_match.id in default_matches
    assert body_only.id not in default_matches  # not scope="all" -- body text not searched

    all_matches = {
        p.id for p in list_postings(db, keyword="distributed systems", keyword_scope="all")
    }
    assert body_only.id in all_matches


def test_list_postings_sort_comp_orders_highest_first(db):
    high = _posting_with_location(
        db, "SortCompHighCo", "https://sortcomphigh.example.com/1", comp_max=300_000
    )
    low = _posting_with_location(
        db, "SortCompLowCo", "https://sortcomplow.example.com/1", comp_max=80_000
    )
    no_comp = _posting_with_location(db, "SortCompNoneCo", "https://sortcompnone.example.com/1")

    # limit=None -- the real dev DB has ~2,744 postings and no deterministic
    # tie-break among NULL-comp rows, so a capped query could leave
    # no_comp out of the result (or reorder ties) unpredictably; uncapped
    # guarantees all three test postings are present to compare.
    ids = [
        p.id
        for p in list_postings(db, sort="comp", limit=None)
        if p.id in {high.id, low.id, no_comp.id}
    ]
    assert ids.index(high.id) < ids.index(low.id)
    # nulls_last -- a posting with no comp data never outranks one that has it.
    assert ids.index(low.id) < ids.index(no_comp.id)


def test_list_postings_sort_relevance_orders_highest_first(db):
    from app.services.relevance import rescore_all

    high = _posting_with_location(db, "SortRelHighCo", "https://sortrelhigh.example.com/1")
    high.normalized_jd_text = "kubernetes kubernetes kubernetes distributed systems"
    low = _posting_with_location(db, "SortRelLowCo", "https://sortrellow.example.com/1")
    low.normalized_jd_text = "sales quota outreach"
    db.flush()
    create_profile_version(
        db,
        snapshot={
            **EMPTY_SNAPSHOT,
            "criteria": {
                **EMPTY_SNAPSHOT["criteria"],
                "target_signals": ["kubernetes distributed systems"],
            },
        },
    )
    rescore_all(db)  # rescores the WHOLE real corpus against this test's profile version

    ids = [
        p.id for p in list_postings(db, sort="relevance", limit=None) if p.id in {high.id, low.id}
    ]
    assert ids.index(high.id) < ids.index(low.id)


def test_list_postings_sort_fit_orders_by_tier_then_relevance(db):
    strong = _posting_with_location(db, "SortFitStrongCo", "https://sortfitstrong.example.com/1")
    unscored = _posting_with_location(
        db, "SortFitUnscoredCo", "https://sortfitunscored.example.com/1"
    )
    skip = _posting_with_location(db, "SortFitSkipCo", "https://sortfitskip.example.com/1")
    strong.verdicts[0].stage3_result = {"verdict": "Strong Fit"}
    skip.verdicts[0].stage1_result = "skip"
    db.flush()

    ids = [
        p.id
        for p in list_postings(db, sort="fit", limit=None)
        if p.id in {strong.id, unscored.id, skip.id}
    ]
    assert ids.index(strong.id) < ids.index(unscored.id) < ids.index(skip.id)


def test_list_postings_invalid_sort_raises_value_error(db):
    with pytest.raises(ValueError):
        list_postings(db, sort="bogus")


def test_count_postings_matches_list_postings_filters(db):
    _posting_with_location(
        db, "CountMatchCo1", "https://countmatch1.example.com/1", workplace_type="remote"
    )
    _posting_with_location(
        db, "CountMatchCo2", "https://countmatch2.example.com/1", workplace_type="onsite"
    )
    assert count_postings(db, workplace_type="remote", company="CountMatch") == 1


def test_unstar_posting_clears_starred_and_starred_at(db):
    posting, _verdict = capture_posting(
        db,
        company_name="UnstarCo",
        title="ML Engineer",
        raw_jd_text="x",
        url="https://unstarco.example.com/1",
    )
    star_posting(db, posting.id)
    assert posting.starred is True
    assert posting.starred_at is not None

    result = unstar_posting(db, posting.id)
    assert result.starred is False
    assert result.starred_at is None


def test_unstar_posting_returns_none_for_unknown_id(db):
    assert unstar_posting(db, 999999) is None


def test_dismiss_and_undismiss_posting(db):
    posting, _verdict = capture_posting(
        db,
        company_name="DismissCo",
        title="ML Engineer",
        raw_jd_text="x",
        url="https://dismissco.example.com/1",
    )
    assert posting.dismissed_at is None

    dismissed = dismiss_posting(db, posting.id)
    assert dismissed.dismissed_at is not None

    undismissed = undismiss_posting(db, posting.id)
    assert undismissed.dismissed_at is None


def test_dismiss_posting_returns_none_for_unknown_id(db):
    assert dismiss_posting(db, 999999) is None
    assert undismiss_posting(db, 999999) is None


def test_set_note_updates_note_without_starring(db):
    posting, _verdict = capture_posting(
        db,
        company_name="NoteCo",
        title="ML Engineer",
        raw_jd_text="x",
        url="https://noteco.example.com/1",
    )
    result = set_note(db, posting.id, "worth a look")
    assert result.note == "worth a look"
    assert result.starred is False

    cleared = set_note(db, posting.id, None)
    assert cleared.note is None


def test_set_note_returns_none_for_unknown_id(db):
    assert set_note(db, 999999, "x") is None


def test_list_stage2_pending_excludes_dismissed_postings(db):
    from app.services.postings import write_stage1_verdict

    company = get_or_create_company(db, "DismissedStage2Co")
    posting, *_ = upsert_posting(
        db,
        company,
        title="Software Engineer II",
        url="https://dismissedstage2.example.com/1",
        external_id=None,
        raw_jd_text="x",
        posted_date=None,
        source="greenhouse_api",
    )
    write_stage1_verdict(db, posting)
    assert posting.id in [p.id for p in list_stage2_pending(db, posting_ids=[posting.id])]

    dismiss_posting(db, posting.id)
    assert posting.id not in [p.id for p in list_stage2_pending(db, posting_ids=[posting.id])]


def test_list_stage3_pending_excludes_dismissed_postings(db):
    posting, _verdict = capture_posting(
        db,
        company_name="DismissedStage3Co",
        title="ML Engineer",
        raw_jd_text="x",
        url="https://dismissedstage3.example.com/1",
    )
    assert posting.id in [p.id for p in list_stage3_pending(db, posting_ids=[posting.id])]

    dismiss_posting(db, posting.id)
    assert posting.id not in [p.id for p in list_stage3_pending(db, posting_ids=[posting.id])]


def test_get_or_create_company_matches_existing_domain(db):
    c1 = get_or_create_company(db, "Acme Inc", url="https://acme.com/careers/123")
    c2 = get_or_create_company(db, "Acme Incorporated", url="https://acme.com/jobs/456")
    assert c1.id == c2.id


def test_list_postings_filters_by_exact_url(db):
    posting, _ = capture_posting(
        db,
        company_name="TestCo6",
        title="ML Engineer",
        raw_jd_text="build models",
        url="https://testco6.example.com/jobs/1?ref=abc",
    )
    matches = list_postings(db, url="https://testco6.example.com/jobs/1?ref=abc")
    assert [p.id for p in matches] == [posting.id]
    assert list_postings(db, url="https://testco6.example.com/jobs/1?ref=different") == []


def test_list_postings_respects_limit(db):
    from app.services.postings import write_stage1_verdict

    company = get_or_create_company(db, "LimitTestCo")
    for i in range(5):
        posting, *_ = upsert_posting(
            db,
            company,
            title="ML Engineer",
            url=None,
            external_id=f"limit-test-{i}",
            raw_jd_text="x",
            posted_date=None,
            source="scraped",
        )
        write_stage1_verdict(db, posting)  # list_postings joins each posting's latest verdict
    assert len(list_postings(db, company="LimitTestCo", limit=3)) == 3
    assert len(list_postings(db, company="LimitTestCo", limit=None)) == 5


def test_list_postings_filters_by_keyword(db):
    posting, _ = capture_posting(
        db,
        company_name="KeywordTestCo",
        title="Staff Machine Learning Engineer",
        raw_jd_text="x",
        url="https://keywordtestco.example.com/1",
    )
    matches = list_postings(db, keyword="machine learning")
    assert posting.id in [p.id for p in matches]
    assert list_postings(db, company="KeywordTestCo", keyword="totally-unrelated-term") == []


def test_count_postings_ignores_limit_and_matches_filters(db):
    from app.services.postings import write_stage1_verdict

    company = get_or_create_company(db, "CountTestCo")
    for i in range(4):
        posting, *_ = upsert_posting(
            db,
            company,
            title="ML Engineer",
            url=None,
            external_id=f"count-test-{i}",
            raw_jd_text="x",
            posted_date=None,
            source="scraped",
        )
        write_stage1_verdict(db, posting)
    assert count_postings(db, company="CountTestCo") == 4
    assert len(list_postings(db, company="CountTestCo", limit=2)) == 2
