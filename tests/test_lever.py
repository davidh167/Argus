"""No Lever company exists in this DB (all real research for the Lever
shapes used here came from live public Lever boards -- see the ro/gopuff
examples referenced in code comments), so every fixture is built from
values confirmed against those live responses rather than DB samples.
"""

from datetime import date
from unittest.mock import patch

from app.crawlers import lever


def _job(**overrides) -> dict:
    base = {
        "id": "abc123",
        "text": "ML Engineer",
        "hostedUrl": "https://jobs.lever.co/testco/abc123",
        "descriptionPlain": "Build things.",
        "lists": [],
        "createdAt": 1755000000000,
        "categories": {
            "location": "New York, NY",
            "allLocations": ["New York, NY"],
            "team": "Data",
        },
        "workplaceType": "hybrid",
        "salaryRange": None,
    }
    base.update(overrides)
    return base


def test_fetch_lever_jobs_maps_workplace_type_and_locations():
    with patch("app.crawlers.lever.httpx.get") as mock_get:
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = [_job()]
        result = lever.fetch_lever_jobs("testco")

    assert result[0].locations == ["New York, NY"]
    assert result[0].workplace_type == "hybrid"
    assert result[0].department == "Data"  # falls back to team when no department


def test_fetch_lever_jobs_prefers_department_over_team():
    # Real gopuff data: categories.department ("BevMo!") is more specific
    # than categories.team ("Leadership") on the same posting.
    job = _job(
        categories={
            "location": "La Cienega, CA",
            "allLocations": ["La Cienega, CA"],
            "team": "Leadership",
            "department": "BevMo!",
        }
    )
    with patch("app.crawlers.lever.httpx.get") as mock_get:
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = [job]
        result = lever.fetch_lever_jobs("testco")

    assert result[0].department == "BevMo!"


def test_fetch_lever_jobs_extracts_salary_range_as_ats_field():
    # Real gopuff shape: {min, max, currency, interval="per-year-salary"}.
    job = _job(
        salaryRange={"min": 75600, "max": 103950, "currency": "USD", "interval": "per-year-salary"}
    )
    with patch("app.crawlers.lever.httpx.get") as mock_get:
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = [job]
        result = lever.fetch_lever_jobs("testco")

    assert result[0].comp_min == 75_600
    assert result[0].comp_max == 103_950
    assert result[0].comp_currency == "USD"
    assert result[0].comp_period == "year"


def test_fetch_lever_jobs_salary_range_none_when_interval_unrecognized():
    job = _job(salaryRange={"min": 20, "max": 30, "currency": "USD", "interval": "some-new-unit"})
    with patch("app.crawlers.lever.httpx.get") as mock_get:
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = [job]
        result = lever.fetch_lever_jobs("testco")

    assert result[0].comp_min is None  # left to upsert_posting's JD-text fallback instead


def test_fetch_lever_jobs_no_salary_range_leaves_comp_unset():
    with patch("app.crawlers.lever.httpx.get") as mock_get:
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = [_job(salaryRange=None)]
        result = lever.fetch_lever_jobs("testco")

    assert result[0].comp_min is None
    assert result[0].comp_currency is None


def test_fetch_lever_jobs_posted_date_from_created_at_ms():
    with patch("app.crawlers.lever.httpx.get") as mock_get:
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = [_job(createdAt=1755000000000)]
        result = lever.fetch_lever_jobs("testco")

    assert result[0].posted_date == date(2025, 8, 12)
