"""Fixtures built from real shapes confirmed live against 4 real Ashby
boards (Acme, Ramp, Ashby, Linear) at
https://api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true
-- not fetched live in tests.
"""

from datetime import date
from unittest.mock import patch

from app.crawlers import ashby


def _job(**overrides) -> dict:
    base = {
        "id": "1fc309c8-da20-4ff2-84c7-8b863ece2b0a",
        "title": "Software Engineer, Developer Platform",
        "department": "Engineering",
        "team": "Engineering",
        "employmentType": "FullTime",
        "location": "San Francisco, California",
        "secondaryLocations": [],
        "publishedAt": "2026-08-24T14:44:49.699+00:00",
        "isListed": True,
        "isRemote": False,
        "workplaceType": "Hybrid",
        "jobUrl": "https://jobs.ashbyhq.com/Acme/1fc309c8-da20-4ff2-84c7-8b863ece2b0a",
        "applyUrl": "https://jobs.ashbyhq.com/Acme/1fc309c8-da20-4ff2-84c7-8b863ece2b0a/application",
        "descriptionPlain": "WHO WE ARE\n\nAcme is the collaborative AI workspace...",
        "compensation": {
            "compensationTierSummary": None,
            "scrapeableCompensationSalarySummary": None,
            "compensationTiers": [],
            "summaryComponents": [],
        },
    }
    base.update(overrides)
    return base


def test_fetch_ashby_jobs_maps_workplace_type_and_locations():
    with patch("app.crawlers.ashby.httpx.get") as mock_get:
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = {"jobs": [_job()], "apiVersion": "5"}
        result = ashby.fetch_ashby_jobs("Acme")

    assert result[0].locations == ["San Francisco, California"]
    assert result[0].workplace_type == "hybrid"
    assert result[0].department == "Engineering"
    assert result[0].url == "https://jobs.ashbyhq.com/Acme/1fc309c8-da20-4ff2-84c7-8b863ece2b0a"


def test_fetch_ashby_jobs_includes_secondary_locations():
    job = _job(
        location="San Francisco, California",
        secondaryLocations=[{"location": "New York, New York"}],
    )
    with patch("app.crawlers.ashby.httpx.get") as mock_get:
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = {"jobs": [job]}
        result = ashby.fetch_ashby_jobs("Acme")

    assert result[0].locations == ["San Francisco, California", "New York, New York"]


def test_fetch_ashby_jobs_onsite_lowercases_correctly():
    # Real Ramp data uses "OnSite" (mixed case) -- confirm .lower() handles
    # it without a mapping table.
    job = _job(workplaceType="OnSite")
    with patch("app.crawlers.ashby.httpx.get") as mock_get:
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = {"jobs": [job]}
        result = ashby.fetch_ashby_jobs("Ramp")

    assert result[0].workplace_type == "onsite"


def test_fetch_ashby_jobs_no_workplace_type_leaves_none():
    job = _job(workplaceType=None)
    with patch("app.crawlers.ashby.httpx.get") as mock_get:
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = {"jobs": [job]}
        result = ashby.fetch_ashby_jobs("Vercel")

    assert result[0].workplace_type is None


def test_fetch_ashby_jobs_extracts_salary_as_ats_field():
    # Real Ramp data: $211.4K - $290.6K.
    job = _job(
        compensation={
            "compensationTierSummary": "$211.4K – $290.6K • Offers Equity",
            "scrapeableCompensationSalarySummary": "$211.4K - $290.6K",
            "compensationTiers": [],
            "summaryComponents": [
                {
                    "compensationType": "EquityPercentage",
                    "interval": "NONE",
                    "currencyCode": None,
                    "minValue": None,
                    "maxValue": None,
                },
                {
                    "compensationType": "Salary",
                    "interval": "1 YEAR",
                    "currencyCode": "USD",
                    "minValue": 211400,
                    "maxValue": 290600,
                },
            ],
        }
    )
    with patch("app.crawlers.ashby.httpx.get") as mock_get:
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = {"jobs": [job]}
        result = ashby.fetch_ashby_jobs("Ramp")

    assert result[0].comp_min == 211_400
    assert result[0].comp_max == 290_600
    assert result[0].comp_currency == "USD"
    assert result[0].comp_period == "year"


def test_fetch_ashby_jobs_no_salary_component_leaves_comp_unset():
    with patch("app.crawlers.ashby.httpx.get") as mock_get:
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = {"jobs": [_job()]}
        result = ashby.fetch_ashby_jobs("Acme")

    assert result[0].comp_min is None
    assert result[0].comp_currency is None


def test_fetch_ashby_jobs_posted_date_from_published_at():
    with patch("app.crawlers.ashby.httpx.get") as mock_get:
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = {
            "jobs": [_job(publishedAt="2026-08-24T14:44:49.699+00:00")]
        }
        result = ashby.fetch_ashby_jobs("Acme")

    assert result[0].posted_date == date(2026, 8, 24)


def test_fetch_ashby_jobs_description_plain_used_directly():
    job = _job(descriptionPlain="Plain text, no HTML tags here.")
    with patch("app.crawlers.ashby.httpx.get") as mock_get:
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = {"jobs": [job]}
        result = ashby.fetch_ashby_jobs("Acme")

    assert result[0].raw_jd_text == "Plain text, no HTML tags here."


def test_fetch_ashby_jobs_handles_empty_jobs_list():
    with patch("app.crawlers.ashby.httpx.get") as mock_get:
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = {"jobs": []}
        result = ashby.fetch_ashby_jobs("EmptyCo")

    assert result == []
