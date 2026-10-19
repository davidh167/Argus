from unittest.mock import patch

import pytest
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.crawlers.base import NormalizedJob
from app.db import engine
from app.models.company import Company
from app.models.verdict import Verdict
from app.services import companies
from app.services.profile import create_profile_version

EMPTY_SNAPSHOT = {
    "experience": [],
    "projects": [],
    "skills": {"core": [], "developing": [], "coursework": []},
    "credentials": {"education": [], "oss_contributions": [], "publications": []},
    "criteria": {
        "skip_patterns": [],
        "target_signals": [],
        "location_prefs": {},
        "seniority_floor": None,
        "seniority_ceiling": None,
    },
}


@pytest.fixture
def db():
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    create_profile_version(session, snapshot=EMPTY_SNAPSHOT)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _make_company(db, name: str) -> "companies.Company":
    from app.models.company import Company

    company = Company(
        name=name,
        ats_type="greenhouse",
        careers_url=f"https://boards.greenhouse.io/{name.lower()}",
    )
    db.add(company)
    db.flush()
    return company


def _verdicts_for(db, posting_id: int) -> list[Verdict]:
    return list(db.scalars(select(Verdict).where(Verdict.posting_id == posting_id)))


def test_recrawl_with_unchanged_title_does_not_stack_a_second_verdict(db):
    company = _make_company(db, "Recrawlco")
    job = NormalizedJob(
        external_id="job-1", title="ML Engineer", url=None, raw_jd_text="v1", posted_date=None
    )
    with patch("app.services.companies.fetch_greenhouse_jobs", return_value=[job]):
        companies.crawl_company(db, company)
    posting = company.postings[0]
    assert len(_verdicts_for(db, posting.id)) == 1

    job_v2 = NormalizedJob(
        external_id="job-1",
        title="ML Engineer",
        url=None,
        raw_jd_text="v2 (edited copy)",
        posted_date=None,
    )
    with patch("app.services.companies.fetch_greenhouse_jobs", return_value=[job_v2]):
        result = companies.crawl_company(db, company)

    assert result["created"] == 0
    assert result["updated"] == 1
    assert len(_verdicts_for(db, posting.id)) == 1  # still exactly one -- the fix
    db.refresh(posting)
    assert posting.raw_jd_text == "v2 (edited copy)"  # text still refreshes


def test_recrawl_preserves_an_existing_stage3_result(db):
    company = _make_company(db, "Preserveco")
    job = NormalizedJob(
        external_id="job-2", title="ML Engineer", url=None, raw_jd_text="v1", posted_date=None
    )
    with patch("app.services.companies.fetch_greenhouse_jobs", return_value=[job]):
        companies.crawl_company(db, company)
    posting = company.postings[0]

    verdict = _verdicts_for(db, posting.id)[0]
    verdict.stage3_result = {"verdict": "Strong Fit"}
    db.flush()

    with patch("app.services.companies.fetch_greenhouse_jobs", return_value=[job]):
        companies.crawl_company(db, company)

    verdicts = _verdicts_for(db, posting.id)
    assert len(verdicts) == 1  # a second write_stage1_verdict would have buried this
    assert verdicts[0].stage3_result == {"verdict": "Strong Fit"}


def test_recrawl_with_changed_title_writes_a_fresh_verdict(db):
    company = _make_company(db, "Titlechangeco")
    job = NormalizedJob(
        external_id="job-3", title="ML Engineer", url=None, raw_jd_text="v1", posted_date=None
    )
    with patch("app.services.companies.fetch_greenhouse_jobs", return_value=[job]):
        companies.crawl_company(db, company)
    posting = company.postings[0]
    assert len(_verdicts_for(db, posting.id)) == 1

    job_retitled = NormalizedJob(
        external_id="job-3",
        title="Senior ML Engineer",
        url=None,
        raw_jd_text="v1",
        posted_date=None,
    )
    with patch("app.services.companies.fetch_greenhouse_jobs", return_value=[job_retitled]):
        companies.crawl_company(db, company)

    verdicts = _verdicts_for(db, posting.id)
    assert len(verdicts) == 2  # title change is a real classify_title input change
    db.refresh(posting)
    assert posting.title == "Senior ML Engineer"


def test_first_crawl_creates_exactly_one_verdict_per_posting(db):
    company = _make_company(db, "Freshco")
    jobs = [
        NormalizedJob(
            external_id="a", title="ML Engineer", url=None, raw_jd_text="x", posted_date=None
        ),
        NormalizedJob(
            external_id="b", title="Data Scientist", url=None, raw_jd_text="y", posted_date=None
        ),
    ]
    with patch("app.services.companies.fetch_greenhouse_jobs", return_value=jobs):
        result = companies.crawl_company(db, company)

    assert result == {
        "jobs_found": 2,
        "created": 2,
        "updated": 0,
        "closed": 0,
        "reopened": 0,
        "closure_evaluated": True,
    }
    for posting in company.postings:
        assert len(_verdicts_for(db, posting.id)) == 1


def test_recrawl_closes_a_posting_that_fell_off_the_board(db):
    company = _make_company(db, "Closeco")
    job_a = NormalizedJob(
        external_id="a", title="ML Engineer", url=None, raw_jd_text="x", posted_date=None
    )
    job_b = NormalizedJob(
        external_id="b", title="Data Scientist", url=None, raw_jd_text="y", posted_date=None
    )
    with patch("app.services.companies.fetch_greenhouse_jobs", return_value=[job_a, job_b]):
        companies.crawl_company(db, company)
    posting_a, posting_b = company.postings[0], company.postings[1]
    assert posting_a.closed_at is None and posting_b.closed_at is None

    # job_b no longer appears on the board.
    with patch("app.services.companies.fetch_greenhouse_jobs", return_value=[job_a]):
        result = companies.crawl_company(db, company)

    assert result["closed"] == 1
    assert result["closure_evaluated"] is True
    db.refresh(posting_a)
    db.refresh(posting_b)
    assert posting_a.closed_at is None  # still on the board
    assert posting_b.closed_at is not None  # fell off


def test_recrawl_reopens_a_previously_closed_posting_and_bumps_repost_count(db):
    company = _make_company(db, "Reopenco")
    job = NormalizedJob(
        external_id="r", title="ML Engineer", url=None, raw_jd_text="x", posted_date=None
    )
    with patch("app.services.companies.fetch_greenhouse_jobs", return_value=[job]):
        companies.crawl_company(db, company)
    posting = company.postings[0]
    assert posting.repost_count == 0

    with patch("app.services.companies.fetch_greenhouse_jobs", return_value=[]):
        empty_result = companies.crawl_company(db, company)
    assert empty_result["closure_evaluated"] is False  # zero-job guard -- see next test
    # Force the close directly to isolate this test from that guard.
    from datetime import datetime, timezone

    posting.closed_at = datetime.now(timezone.utc)
    db.flush()

    with patch("app.services.companies.fetch_greenhouse_jobs", return_value=[job]):
        result = companies.crawl_company(db, company)

    assert result["reopened"] == 1
    db.refresh(posting)
    assert posting.closed_at is None
    assert posting.repost_count == 1


def test_recrawl_with_zero_jobs_skips_closure(db):
    company = _make_company(db, "Zerojobsco")
    job = NormalizedJob(
        external_id="z", title="ML Engineer", url=None, raw_jd_text="x", posted_date=None
    )
    with patch("app.services.companies.fetch_greenhouse_jobs", return_value=[job]):
        companies.crawl_company(db, company)
    posting = company.postings[0]

    # A broken fetch returning nothing must never look like "the whole
    # board closed."
    with patch("app.services.companies.fetch_greenhouse_jobs", return_value=[]):
        result = companies.crawl_company(db, company)

    assert result == {
        "jobs_found": 0,
        "created": 0,
        "updated": 0,
        "closed": 0,
        "reopened": 0,
        "closure_evaluated": False,
    }
    db.refresh(posting)
    assert posting.closed_at is None


def test_recrawl_skips_closure_when_careers_url_changed(db):
    company = _make_company(db, "Urlchangeco")
    job_a = NormalizedJob(
        external_id="a", title="ML Engineer", url=None, raw_jd_text="x", posted_date=None
    )
    job_b = NormalizedJob(
        external_id="b", title="Data Scientist", url=None, raw_jd_text="y", posted_date=None
    )
    with patch("app.services.companies.fetch_greenhouse_jobs", return_value=[job_a, job_b]):
        companies.crawl_company(db, company)
    posting_b = company.postings[1]
    assert company.last_crawl_url == company.careers_url

    # The user points this company at a different (e.g. filtered) board URL.
    company.careers_url = company.careers_url + "?department=engineering"
    db.flush()

    # This crawl of the NEW url only returns job_a -- if closure were
    # evaluated (wrongly, since this doesn't cover the same set as before),
    # posting_b would be closed. It must not be.
    with patch("app.services.companies.fetch_greenhouse_jobs", return_value=[job_a]):
        result = companies.crawl_company(db, company)

    assert result["closure_evaluated"] is False
    db.refresh(posting_b)
    assert posting_b.closed_at is None
    assert company.last_crawl_url == company.careers_url  # recorded for next time


def _make_scrape_company(db, name: str) -> "companies.Company":
    company = Company(
        name=name,
        ats_type=None,
        careers_url=f"https://{name.lower()}.example.com/careers",
        domain=f"{name.lower()}.example.com",
    )
    db.add(company)
    db.flush()
    return company


def _disable_all_existing_companies(db):
    """refresh_all_companies has no id-scoping -- it loops over every
    crawl_enabled=True company with no filter, and this test DB fixture's
    rollback-only isolation does NOT stop it from seeing (and acting on,
    with real network calls) already-committed real companies in the
    shared dev DB. Confirmed the hard way: an earlier, unguarded version
    of these tests fired a real Greenhouse GET and a real Apple scrape
    attempt. Neutralize every pre-existing real company within this
    test's own transaction (rolled back at teardown either way) before
    calling refresh_all_companies, so it can only ever touch this test's
    own fixtures."""
    db.execute(update(Company).values(crawl_enabled=False))
    db.flush()


def test_refresh_all_companies_skips_ones_with_no_careers_url(db):
    _disable_all_existing_companies(db)
    no_url_co = Company(name="NoUrlCo", careers_url=None, crawl_enabled=True)
    db.add(no_url_co)
    db.flush()

    results = companies.refresh_all_companies(db)
    assert results == []


def test_refresh_all_companies_attempts_crawl_and_scrape_companies(db):
    _disable_all_existing_companies(db)
    crawl_co = _make_company(db, "RefreshCrawlCo")
    scrape_co = _make_scrape_company(db, "RefreshScrapeCo")

    job = NormalizedJob(
        external_id="1", title="ML Engineer", url=None, raw_jd_text="x", posted_date=None
    )
    with patch("app.services.companies.fetch_greenhouse_jobs", return_value=[job]):
        with patch("app.services.scraper_catalog.scrape_company") as mock_scrape:
            mock_scrape.return_value = {
                "jobs_found": 1,
                "created": 1,
                "updated": 0,
                "closed": 0,
                "reopened": 0,
                "fallback_used": False,
                "closure_evaluated": True,
                "hydrated_inline": 0,
            }
            results = companies.refresh_all_companies(db)

    by_id = {r["company_id"]: r for r in results}
    assert by_id[crawl_co.id]["ok"] is True
    assert by_id[crawl_co.id]["result"]["created"] == 1
    assert by_id[scrape_co.id]["ok"] is True
    assert by_id[scrape_co.id]["result"]["created"] == 1
    mock_scrape.assert_called_once()


def test_refresh_all_companies_records_per_company_failure_and_continues(db):
    # A company missing its slug-extractable careers_url shape (e.g. a
    # bare domain with no path) is exactly what UnsupportedAtsError covers
    # in real use -- extract_slug returns None, crawl_company raises.
    _disable_all_existing_companies(db)
    failing_co = _make_company(db, "RefreshFailCo")
    failing_co.careers_url = "https://boards.greenhouse.io/"  # no slug segment
    ok_co = _make_company(db, "RefreshOkCo")
    db.flush()

    job = NormalizedJob(
        external_id="1", title="ML Engineer", url=None, raw_jd_text="x", posted_date=None
    )
    with patch("app.services.companies.fetch_greenhouse_jobs", return_value=[job]):
        results = companies.refresh_all_companies(db)

    by_id = {r["company_id"]: r for r in results}
    assert by_id[failing_co.id]["ok"] is False
    assert "slug" in by_id[failing_co.id]["error"].lower()
    # The loop kept going past the failure -- ok_co was still attempted.
    assert by_id[ok_co.id]["ok"] is True


def test_extract_domain_plain_single_tenant_host():
    assert companies.extract_domain("https://jobs.apple.com/en-us/details/123") == "jobs.apple.com"


def test_extract_domain_strips_www():
    assert companies.extract_domain("https://www.example.com/jobs/1") == "example.com"


def test_extract_domain_multi_tenant_ats_host_includes_slug():
    # Regression test: many unrelated
    # companies share the same Greenhouse/Lever/Ashby hostname, so the
    # hostname alone must not be treated as a company identity.
    assert (
        companies.extract_domain("https://jobs.ashbyhq.com/acme/abc-123") == "jobs.ashbyhq.com/acme"
    )
    assert (
        companies.extract_domain("https://job-boards.greenhouse.io/globex/456")
        == "job-boards.greenhouse.io/globex"
    )


def test_extract_domain_multi_tenant_host_with_no_slug_is_none():
    assert companies.extract_domain("https://jobs.ashbyhq.com/") is None


def test_get_or_create_company_does_not_merge_different_companies_on_shared_ats_host(db):
    # The actual real-world failure: two different companies' postings on
    # the same multi-tenant ATS host used to collide into one Company row.
    # Fake, distinctive names -- real company names in the shared dev DB
    # (like an existing company row) would collide via the name-fallback
    # match and mask what this test is checking.
    co1 = companies.get_or_create_company(
        db, "SlugTestCoOne", "https://jobs.ashbyhq.com/slugtestcoone/1"
    )
    co2 = companies.get_or_create_company(
        db, "SlugTestCoTwo", "https://jobs.ashbyhq.com/slugtestcotwo/2"
    )
    assert co1.id != co2.id
    assert co1.domain == "jobs.ashbyhq.com/slugtestcoone"
    assert co2.domain == "jobs.ashbyhq.com/slugtestcotwo"


def test_get_or_create_company_still_matches_the_same_company_by_slug(db):
    first = companies.get_or_create_company(
        db, "SlugTestCoThree", "https://jobs.ashbyhq.com/slugtestcothree/1"
    )
    second = companies.get_or_create_company(
        db, "SlugTestCoThree", "https://jobs.ashbyhq.com/slugtestcothree/2"
    )
    assert first.id == second.id
