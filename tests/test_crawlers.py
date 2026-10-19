from datetime import date
from unittest.mock import MagicMock, patch

from app.crawlers import greenhouse
from app.crawlers.base import clean_html_text


def test_clean_html_text_unescapes_and_converts_block_tags_to_newlines():
    # Shaped like Greenhouse's real `content` field: HTML that is itself
    # HTML-entity-escaped, not literal markup.
    raw = (
        "&lt;div class=&quot;content-intro&quot;&gt;&lt;h2&gt;About Us&lt;/h2&gt;"
        "&lt;p&gt;We build things.&lt;/p&gt;&lt;/div&gt;"
    )
    cleaned = clean_html_text(raw)
    assert "&lt;" not in cleaned
    assert "&quot;" not in cleaned
    assert "About Us" in cleaned
    assert "We build things." in cleaned
    assert "\n" in cleaned  # the </h2>/</p>/</div> boundaries became newlines


def test_clean_html_text_handles_a_pay_range_block_like_the_real_data():
    # Shaped like Anthropic's real postings: an escaped pay-range widget
    # embedded mid-paragraph.
    raw = (
        "&lt;p&gt;The starting salary will vary based on factors including "
        "location, experience.&lt;/p&gt;&lt;div class=&quot;title&quot;&gt;"
        "Annual Salary:&lt;/div&gt;&lt;div class=&quot;pay-range&quot;&gt;"
        "&lt;span&gt;$270,000&lt;/span&gt;&lt;span class=&quot;divider&quot;&gt;"
        "&amp;mdash;&lt;/span&gt;&lt;span&gt;$340,000 USD&lt;/span&gt;&lt;/div&gt;"
    )
    cleaned = clean_html_text(raw)
    assert "&lt;" not in cleaned and "&gt;" not in cleaned
    assert "$270,000" in cleaned
    assert "$340,000 USD" in cleaned
    assert "Annual Salary:" in cleaned
    # Real bug found via live data: &amp;mdash; is double-escaped (the
    # divider itself, not the surrounding tags) -- a single unescape pass
    # leaves the literal text "&mdash;" sitting between the two figures
    # instead of a real em-dash, which silently breaks any attempt to
    # parse "$270,000 -- $340,000" as a range.
    assert "&mdash;" not in cleaned
    assert "$270,000 — $340,000 USD" in cleaned


def test_clean_html_text_resolves_double_escaped_entities():
    # &amp;nbsp; (double-escaped) must become a real non-breaking space, not
    # the literal text "&nbsp;" -- same double-escaping as &amp;mdash;.
    raw = "&lt;p&gt;partners,&amp;nbsp; including vendors.&lt;/p&gt;"
    cleaned = clean_html_text(raw)
    assert "&nbsp;" not in cleaned
    assert "\xa0" in cleaned


def test_clean_html_text_converts_br_tags():
    raw = "Line one&lt;br&gt;Line two&lt;br/&gt;Line three"
    cleaned = clean_html_text(raw)
    assert cleaned.count("\n") == 2
    assert "Line one" in cleaned and "Line two" in cleaned and "Line three" in cleaned


def test_clean_html_text_collapses_whitespace_runs():
    raw = "&lt;p&gt;a&lt;/p&gt;&lt;p&gt;&lt;/p&gt;&lt;p&gt;b&lt;/p&gt;"
    cleaned = clean_html_text(raw)
    assert "\n\n" not in cleaned  # runs of newlines collapse to one
    assert "  " not in cleaned  # runs of spaces collapse to one


def test_clean_html_text_is_a_noop_on_plain_text():
    assert clean_html_text("Just plain text, no markup.") == "Just plain text, no markup."


def test_clean_html_text_handles_none_and_empty():
    assert clean_html_text(None) is None
    assert clean_html_text("") == ""


def _mock_response(jobs: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"jobs": jobs}
    return resp


def test_fetch_greenhouse_jobs_prefers_first_published_over_updated_at():
    jobs = [
        {
            "id": 1,
            "title": "ML Engineer",
            "absolute_url": "https://example.com/1",
            "content": "&lt;p&gt;Build things.&lt;/p&gt;",
            "first_published": "2026-08-01T00:00:00-04:00",
            "updated_at": "2026-09-17T00:00:00-04:00",  # a later edit, not the real post date
        }
    ]
    with patch("app.crawlers.greenhouse.httpx.get", return_value=_mock_response(jobs)):
        result = greenhouse.fetch_greenhouse_jobs("testco")

    assert result[0].posted_date == date(2026, 8, 1)
    assert "Build things." in result[0].raw_jd_text
    assert "&lt;" not in result[0].raw_jd_text


def test_fetch_greenhouse_jobs_falls_back_to_updated_at_when_first_published_missing():
    jobs = [
        {
            "id": 2,
            "title": "Data Scientist",
            "absolute_url": "https://example.com/2",
            "content": "text",
            "updated_at": "2026-09-17T00:00:00-04:00",
        }
    ]
    with patch("app.crawlers.greenhouse.httpx.get", return_value=_mock_response(jobs)):
        result = greenhouse.fetch_greenhouse_jobs("testco")

    assert result[0].posted_date == date(2026, 9, 17)


def test_fetch_greenhouse_jobs_posted_date_none_when_both_missing():
    jobs = [{"id": 3, "title": "X", "absolute_url": None, "content": None}]
    with patch("app.crawlers.greenhouse.httpx.get", return_value=_mock_response(jobs)):
        result = greenhouse.fetch_greenhouse_jobs("testco")

    assert result[0].posted_date is None


def _job_with_metadata(location_type: str | None, location_name: str, department: str | None):
    return {
        "id": 10,
        "title": "ML Engineer",
        "absolute_url": "https://example.com/10",
        "content": "text",
        "location": {"name": location_name},
        "departments": [{"name": department}] if department else [],
        "metadata": [{"name": "Location Type", "value": location_type}] if location_type else [],
    }


def test_fetch_greenhouse_jobs_maps_location_type_and_splits_locations():
    # Shaped after Anthropic's real values.
    jobs = [_job_with_metadata("On-Site", "San Francisco, CA | New York City, NY", "Engineering")]
    with patch("app.crawlers.greenhouse.httpx.get", return_value=_mock_response(jobs)):
        result = greenhouse.fetch_greenhouse_jobs("testco")

    assert result[0].locations == ["San Francisco, CA", "New York City, NY"]
    assert result[0].workplace_type == "onsite"
    assert result[0].department == "Engineering"


def test_fetch_greenhouse_jobs_maps_remote_and_hybrid_travel_required():
    jobs = [
        _job_with_metadata("Remote", "Remote-Friendly, United States", "Sales"),
        {**_job_with_metadata("Hybrid (Travel-Required)", "Seattle, WA", "Sales"), "id": 11},
    ]
    with patch("app.crawlers.greenhouse.httpx.get", return_value=_mock_response(jobs)):
        result = greenhouse.fetch_greenhouse_jobs("testco")

    assert result[0].workplace_type == "remote"
    assert result[1].workplace_type == "hybrid"


def test_fetch_greenhouse_jobs_infers_remote_when_location_type_metadata_missing():
    # ~9% of Anthropic's live board has no "Location Type" metadata at all.
    jobs = [_job_with_metadata(None, "Remote-Friendly, United States", "Sales")]
    with patch("app.crawlers.greenhouse.httpx.get", return_value=_mock_response(jobs)):
        result = greenhouse.fetch_greenhouse_jobs("testco")

    assert result[0].workplace_type == "remote"


def test_fetch_greenhouse_jobs_workplace_type_none_when_unrecoverable():
    jobs = [
        _job_with_metadata(None, "San Francisco, CA", "Sales")
    ]  # no metadata, not remote-labeled
    with patch("app.crawlers.greenhouse.httpx.get", return_value=_mock_response(jobs)):
        result = greenhouse.fetch_greenhouse_jobs("testco")

    assert result[0].workplace_type is None


def test_fetch_greenhouse_jobs_extracts_comp_from_cleaned_content():
    jobs = [
        {
            "id": 12,
            "title": "ML Engineer",
            "absolute_url": "https://example.com/12",
            "content": "&lt;p&gt;Annual Salary: $270,000&amp;mdash; $340,000 USD&lt;/p&gt;",
            "location": {"name": "San Francisco, CA"},
            "departments": [],
            "metadata": [],
        }
    ]
    with patch("app.crawlers.greenhouse.httpx.get", return_value=_mock_response(jobs)):
        result = greenhouse.fetch_greenhouse_jobs("testco")

    # Greenhouse itself has no comp field -- NormalizedJob's comp fields
    # stay unset here; extraction happens once, uniformly, inside
    # postings.upsert_posting (see tests/test_postings_service.py).
    assert result[0].comp_min is None
    assert "270,000" in result[0].raw_jd_text
