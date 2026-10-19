from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from app.crawlers.fetch import FetchStrategy
from app.db import engine
from app.models.company import Company
from app.models.site_scraper import SiteScraper
from app.services import scraper_catalog

FIXTURE_HTML = """
<html><body>
<div class="jobs-list">
  <div class="promo">Follow us on Twitter!</div>
  <div class="job-card">
    <a class="job-title" href="/jobs/123">Senior ML Engineer</a>
    <div class="job-desc">Build ranking systems.</div>
    <span class="job-date">2026-09-01</span>
  </div>
  <div class="job-card">
    <a class="job-title" href="/jobs/124">Data Scientist</a>
    <div class="job-desc">Analyze data.</div>
    <span class="job-date">2026-08-15</span>
  </div>
</div>
</body></html>
"""

FIXTURE_STRATEGY = {
    "type": "css_selector",
    "listings_container": "div.job-card",
    "title_selector": "a.job-title",
    "description_selector": "div.job-desc",
    "posted_date_selector": "span.job-date",
    "pagination_pattern": "",
    "version": 1,
}


def test_apply_extraction_strategy_extracts_fields_and_resolves_relative_urls():
    results = scraper_catalog.apply_extraction_strategy(
        FIXTURE_HTML, FIXTURE_STRATEGY, base_url="https://jobs.example.com/careers"
    )
    assert len(results) == 2  # the .promo div is skipped -- div.job-card doesn't match it
    assert results[0]["title"] == "Senior ML Engineer"
    assert results[0]["url"] == "https://jobs.example.com/jobs/123"
    assert results[0]["description"] == "Build ranking systems."
    assert results[1]["url"] == "https://jobs.example.com/jobs/124"


def test_apply_extraction_strategy_raises_on_unsupported_type():
    bad_strategy = {**FIXTURE_STRATEGY, "type": "xpath"}
    with pytest.raises(scraper_catalog.UnsupportedStrategyTypeError):
        scraper_catalog.apply_extraction_strategy(
            FIXTURE_HTML, bad_strategy, base_url="https://x.com"
        )


def test_validate_extraction_requires_title_and_url():
    assert scraper_catalog.validate_extraction([]) is False
    assert scraper_catalog.validate_extraction([{"title": "X", "url": None}]) is False
    assert scraper_catalog.validate_extraction([{"title": "X", "url": "https://x.com/1"}]) is True


def test_build_scraper_user_message_wraps_html_in_delimiter_tags():
    msg = scraper_catalog.build_scraper_user_message(FIXTURE_HTML)
    assert msg.count("<career_page_html>") == 1
    assert msg.count("</career_page_html>") == 1
    start = msg.index("<career_page_html>")
    end = msg.index("</career_page_html>")
    assert start < msg.index("Senior ML Engineer") < end


def test_generate_extraction_strategy_raises_stage2_unavailable_without_api_key():
    with patch.object(scraper_catalog.settings, "anthropic_api_key", None):
        with patch("app.services.scraper_catalog.Anthropic") as mock_client_cls:
            with pytest.raises(scraper_catalog.Stage2UnavailableError):
                scraper_catalog.generate_extraction_strategy(FIXTURE_HTML)
            mock_client_cls.assert_not_called()


def test_generate_extraction_strategy_parses_tool_use_response():
    fake_tool_use_block = MagicMock(type="tool_use", input=dict(FIXTURE_STRATEGY))
    fake_response = MagicMock(content=[fake_tool_use_block])
    with patch.object(scraper_catalog.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.scraper_catalog.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
            mock_client_cls.return_value = mock_client

            result = scraper_catalog.generate_extraction_strategy(FIXTURE_HTML)

    assert result["type"] == "css_selector"
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "generate_extraction_strategy"}
    assert call_kwargs["model"] == scraper_catalog.SCRAPER_MODEL


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Sep 22, 2026", date(2026, 9, 22)),
        ("September 22, 2026", date(2026, 9, 22)),
        ("2026-09-22", date(2026, 9, 22)),
    ],
)
def test_parse_posted_date_handles_known_formats(text, expected):
    assert scraper_catalog.parse_posted_date(text) == expected


def test_parse_posted_date_returns_none_for_unparseable_text(caplog):
    assert scraper_catalog.parse_posted_date("3 days ago") is None
    assert "Could not parse" in caplog.text


@pytest.mark.parametrize("text", [None, ""])
def test_parse_posted_date_returns_none_for_none_or_empty(text):
    assert scraper_catalog.parse_posted_date(text) is None


def _job_card_html(page: int, count: int) -> str:
    cards = "\n".join(
        f'<div class="job-card"><a class="job-title" href="/jobs/{page}-{i}">Job {page}-{i}</a>'
        f'<span class="job-date">2026-09-{min(page, 28):02d}</span></div>'
        for i in range(count)
    )
    return f"<html><body><div class='jobs-list'>{cards}</div></body></html>"


PAGINATED_STRATEGY = {
    "type": "css_selector",
    "listings_container": "div.job-card",
    "title_selector": "a.job-title",
    "description_selector": "",
    "posted_date_selector": "span.job-date",
    "pagination_pattern": "?page=N query param",
    "paginated": True,
    "page_query_param": "page",
    "version": 1,
}


class FakeFetch(FetchStrategy):
    """Serves canned HTML per ?page=N, empty for any page not given."""

    def __init__(self, pages: dict[int, str]):
        self.pages = pages
        self.calls: list[str] = []

    def fetch(self, url: str) -> str:
        self.calls.append(url)
        from urllib.parse import parse_qs, urlsplit

        qs = parse_qs(urlsplit(url).query)
        page = int(qs.get("page", ["1"])[0])
        return self.pages.get(page, "<html><body></body></html>")


def _listing_calls(fetch: FakeFetch, careers_url: str) -> list[str]:
    """scrape_company also fetches each newly created posting's own URL
    for inline hydration (capped at SCRAPE_INLINE_HYDRATE_CAP),
    through this same fetch_strategy -- filters those out so pagination
    tests can assert on listing/pagination calls specifically."""
    return [c for c in fetch.calls if c == careers_url or "page=" in c]


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
    from app.services.profile import create_profile_version

    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    create_profile_version(session, snapshot=EMPTY_SNAPSHOT)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _make_company(db, domain: str) -> Company:
    company = Company(
        name="PaginationTestCo", domain=domain, careers_url=f"https://{domain}/careers"
    )
    db.add(company)
    db.flush()
    return company


def _seed_scraper(db, domain: str, strategy: dict) -> None:
    db.add(SiteScraper(domain=domain, extraction_strategy=strategy))
    db.flush()


def test_scrape_company_paginates_until_empty_page(db):
    domain = "paginated1.example.com"
    company = _make_company(db, domain)
    _seed_scraper(db, domain, PAGINATED_STRATEGY)
    fetch = FakeFetch({1: _job_card_html(1, 2), 2: _job_card_html(2, 2), 3: _job_card_html(3, 2)})

    result = scraper_catalog.scrape_company(db, company, fetch_strategy=fetch)

    assert (
        len(_listing_calls(fetch, company.careers_url)) == 4
    )  # pages 1-3 with content, page 4 empty stops the loop
    assert result["jobs_found"] == 6
    assert result["created"] == 6
    assert result["closure_evaluated"] is True  # pagination ran to its natural end


def test_scrape_company_respects_safety_cap(db, monkeypatch):
    domain = "paginated2.example.com"
    company = _make_company(db, domain)
    _seed_scraper(db, domain, PAGINATED_STRATEGY)
    monkeypatch.setattr(scraper_catalog, "SCRAPE_MAX_PAGES", 3)
    # Every page has content -- without the cap this would loop forever.
    fetch = FakeFetch({p: _job_card_html(p, 1) for p in range(1, 50)})

    result = scraper_catalog.scrape_company(db, company, fetch_strategy=fetch)

    assert len(_listing_calls(fetch, company.careers_url)) == 3
    assert result["jobs_found"] == 3
    # The cap was hit while pages still had content -- this scrape may not
    # cover the whole board, so closure marking must be skipped.
    assert result["closure_evaluated"] is False


def test_scrape_company_skips_pagination_when_not_paginated(db):
    domain = "notpaginated.example.com"
    company = _make_company(db, domain)
    strategy = {**PAGINATED_STRATEGY, "paginated": False}
    _seed_scraper(db, domain, strategy)
    fetch = FakeFetch({1: _job_card_html(1, 2), 2: _job_card_html(2, 2)})

    result = scraper_catalog.scrape_company(db, company, fetch_strategy=fetch)

    assert len(_listing_calls(fetch, company.careers_url)) == 1
    assert result["jobs_found"] == 2


def test_scrape_company_persists_posted_date(db):
    domain = "dates.example.com"
    company = _make_company(db, domain)
    strategy = {**PAGINATED_STRATEGY, "paginated": False}
    _seed_scraper(db, domain, strategy)
    fetch = FakeFetch({1: _job_card_html(1, 1)})

    scraper_catalog.scrape_company(db, company, fetch_strategy=fetch)

    posting = company.postings[0]
    assert posting.posted_date == date(2026, 9, 1)


_DETAIL_PAGE_HTML = (
    "<html><body><h1>Full Job Detail Page</h1>"
    "<p>About the Role: This is the full job description fetched from the "
    "posting's own detail page URL, much longer than the short listing "
    "blurb, containing everything a candidate would need to know about "
    "this specific role and team.</p></body></html>"
)


class FakeFetchWithDetail(FetchStrategy):
    """Serves listing/pagination pages by ?page=N (like FakeFetch), and a
    fixed detail-page HTML for any other URL (a posting's own detail
    page) -- long enough to succeed via detail_fetch's visible-text tier."""

    def __init__(self, pages: dict[int, str]):
        self.pages = pages
        self.calls: list[str] = []

    def fetch(self, url: str) -> str:
        self.calls.append(url)
        from urllib.parse import parse_qs, urlsplit

        if "/jobs/" in url:  # a posting's own detail page, not a listing page
            return _DETAIL_PAGE_HTML
        qs = parse_qs(urlsplit(url).query)
        page = int(qs.get("page", ["1"])[0])
        return self.pages.get(page, "<html><body></body></html>")


def test_scrape_company_hydrates_new_postings_inline_up_to_cap(db, monkeypatch):
    monkeypatch.setattr(scraper_catalog, "SCRAPE_INLINE_HYDRATE_CAP", 2)
    domain = "inlinehydrate.example.com"
    company = _make_company(db, domain)
    strategy = {**PAGINATED_STRATEGY, "paginated": False}
    _seed_scraper(db, domain, strategy)
    fetch = FakeFetchWithDetail({1: _job_card_html(1, 3)})  # 3 new postings, cap is 2

    result = scraper_catalog.scrape_company(db, company, fetch_strategy=fetch)

    assert result["created"] == 3
    assert result["hydrated_inline"] == 2  # capped, not all 3
    hydrated_count = sum(1 for p in company.postings if p.detail_fetched_at is not None)
    assert hydrated_count == 2
    for posting in company.postings:
        if posting.detail_fetched_at is not None:
            assert "full job description" in posting.raw_jd_text.lower()


def test_rescrape_does_not_hydrate_inline_for_already_seen_postings(db):
    domain = "norehydrate.example.com"
    company = _make_company(db, domain)
    strategy = {**PAGINATED_STRATEGY, "paginated": False}
    _seed_scraper(db, domain, strategy)
    fetch = FakeFetchWithDetail({1: _job_card_html(1, 1)})

    scraper_catalog.scrape_company(db, company, fetch_strategy=fetch)
    posting = company.postings[0]
    assert posting.detail_fetched_at is not None  # hydrated on first (creation) scrape
    first_hydrated_at = posting.detail_fetched_at
    calls_after_first_scrape = len(fetch.calls)

    result = scraper_catalog.scrape_company(db, company, fetch_strategy=fetch)

    assert result["created"] == 0  # nothing new this time
    assert result["hydrated_inline"] == 0
    db.refresh(posting)
    assert posting.detail_fetched_at == first_hydrated_at  # untouched
    # No extra detail-page fetch for the already-hydrated posting -- only
    # the listing page itself was re-fetched.
    assert len(fetch.calls) == calls_after_first_scrape + 1


def test_rescrape_with_unchanged_titles_does_not_stack_verdicts(db):
    from sqlalchemy import select

    from app.models.verdict import Verdict

    domain = "rescrape.example.com"
    company = _make_company(db, domain)
    strategy = {**PAGINATED_STRATEGY, "paginated": False}
    _seed_scraper(db, domain, strategy)
    fetch = FakeFetch({1: _job_card_html(1, 2)})

    scraper_catalog.scrape_company(db, company, fetch_strategy=fetch)
    posting_ids = [p.id for p in company.postings]
    assert len(posting_ids) == 2
    for pid in posting_ids:
        assert (
            db.scalar(select(Verdict).where(Verdict.posting_id == pid).order_by(Verdict.id))
            is not None
        )

    def _verdict_count(pid):
        return len(list(db.scalars(select(Verdict).where(Verdict.posting_id == pid))))

    assert all(_verdict_count(pid) == 1 for pid in posting_ids)

    # Same page fetched again -- titles unchanged, so no new verdicts.
    result = scraper_catalog.scrape_company(db, company, fetch_strategy=fetch)

    assert result["created"] == 0
    assert result["updated"] == 2
    assert all(_verdict_count(pid) == 1 for pid in posting_ids)


def test_rescrape_closes_a_posting_that_fell_off_the_page(db):
    domain = "rescrape-close.example.com"
    company = _make_company(db, domain)
    strategy = {**PAGINATED_STRATEGY, "paginated": False}
    _seed_scraper(db, domain, strategy)

    scraper_catalog.scrape_company(db, company, fetch_strategy=FakeFetch({1: _job_card_html(1, 2)}))
    posting_a, posting_b = company.postings[0], company.postings[1]
    assert posting_a.closed_at is None and posting_b.closed_at is None

    # Second job card no longer listed.
    result = scraper_catalog.scrape_company(
        db, company, fetch_strategy=FakeFetch({1: _job_card_html(1, 1)})
    )

    assert result["closed"] == 1
    assert result["closure_evaluated"] is True
    db.refresh(posting_a)
    db.refresh(posting_b)
    assert posting_a.closed_at is None
    assert posting_b.closed_at is not None


def test_rescrape_hitting_the_page_cap_skips_closure(db, monkeypatch):
    domain = "rescrape-cap.example.com"
    company = _make_company(db, domain)
    _seed_scraper(db, domain, PAGINATED_STRATEGY)
    monkeypatch.setattr(scraper_catalog, "SCRAPE_MAX_PAGES", 2)

    # First scrape (page 1 has 2 cards: /jobs/1-0, /jobs/1-1; page 2 is
    # naturally empty, so this one completes within the cap).
    scraper_catalog.scrape_company(db, company, fetch_strategy=FakeFetch({1: _job_card_html(1, 2)}))
    posting_seen_again, posting_off_the_capped_pages = company.postings[0], company.postings[1]

    # Second scrape: every page has content, so it hits SCRAPE_MAX_PAGES=2
    # without a natural empty-page stop. Page 1 here is /jobs/1-0 again
    # (still "seen"), but /jobs/1-1 never appears on any page this time --
    # if closure were (wrongly) evaluated, it would be marked closed.
    every_page_has_content = FakeFetch({p: _job_card_html(p, 1) for p in range(1, 50)})
    result = scraper_catalog.scrape_company(db, company, fetch_strategy=every_page_has_content)

    assert result["closure_evaluated"] is False
    db.refresh(posting_seen_again)
    db.refresh(posting_off_the_capped_pages)
    assert posting_seen_again.closed_at is None
    assert posting_off_the_capped_pages.closed_at is None  # not closed -- the cap, not absence
