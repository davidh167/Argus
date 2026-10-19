"""Every fixture here is synthetic (trimmed/hand-built), not fetched live,
per the plan's explicit instruction -- but the Apple hydration shape and
field names are calibrated against real live fetches (see
app/services/detail_fetch.py's module docstring for exactly which real
Apple job pages were used to confirm them).
"""

import json

from app.crawlers.fetch import FetchStrategy
from app.services.detail_fetch import extract_detail, fetch_and_extract_detail

# ---- Apple-shaped hydration JSON ----


def _apple_html(job_data: dict) -> str:
    """Wraps job_data exactly as Apple's real page does: a JS statement
    assigning JSON.parse("<JS-string-escaped JSON>") inside a <script>
    tag with no distinguishing type/id attribute."""
    payload = {"loaderData": {"jobDetails": {"jobsData": job_data}}}
    json_text = json.dumps(payload)
    js_string_literal = json.dumps(json_text)  # adds the JS-escaped outer quotes
    return (
        "<html><head></head><body>"
        f'<script nonce="abc">window.__staticRouterHydrationData = '
        f"JSON.parse({js_string_literal});</script>"
        "</body></html>"
    )


APPLE_JOB = {
    "postingTitle": "Thermal Engineer",
    "jobSummary": "The Apple Thermal team is seeking a world-class Thermal Engineer.",
    "description": "You will own thermal design for upcoming products.",
    "responsibilities": "Design heat sinks. Run thermal simulations. Validate prototypes.",
    "minimumQualifications": "BS in Mechanical Engineering. 5+ years experience.",
    "preferredQualifications": "MS or PhD preferred. Experience with vapor chambers.",
    "postDateInGMT": "2026-09-23T18:32:46.599+00:00",
    "locations": [
        {"city": "Austin", "stateProvince": "Texas", "countryName": "United States"},
        {"city": "Cupertino", "stateProvince": "California", "countryName": "United States"},
    ],
    "postingFooters": [
        {
            "localizations": {
                "en_US": [
                    {
                        "name": "Pay & Benefits",
                        "content": (
                            "The base pay range for this role is between $150,400 "
                            "and $277,600, and your base pay will depend on your skills."
                        ),
                    }
                ]
            }
        }
    ],
}


def test_apple_hydration_extracts_full_text_locations_and_date():
    html = _apple_html(APPLE_JOB)
    result = extract_detail(html)

    assert result is not None
    assert "world-class Thermal Engineer" in result.raw_jd_text
    assert "Design heat sinks" in result.raw_jd_text
    assert "BS in Mechanical Engineering" in result.raw_jd_text
    assert "MS or PhD preferred" in result.raw_jd_text
    assert result.posted_date.isoformat() == "2026-09-23"
    assert result.locations == ["Austin, Texas", "Cupertino, California"]


def test_apple_hydration_folds_pay_and_benefits_text_into_raw_jd_text():
    # Not extracted as its own field -- it must survive into raw_jd_text so
    # the generic JD-text comp extractor (run by upsert_posting) finds it.
    html = _apple_html(APPLE_JOB)
    result = extract_detail(html)
    assert "$150,400" in result.raw_jd_text
    assert "$277,600" in result.raw_jd_text


def test_apple_hydration_sections_are_findable_by_jd_sections():
    from app.services.jd_sections import extract_jd_sections

    html = _apple_html(APPLE_JOB)
    result = extract_detail(html)
    sections = extract_jd_sections(result.raw_jd_text)
    assert "Design heat sinks" in sections.responsibilities
    assert "BS in Mechanical Engineering" in sections.required_qualifications
    assert "MS or PhD preferred" in sections.preferred_qualifications


def test_apple_hydration_tolerant_of_missing_optional_fields():
    minimal = {
        "postingTitle": "Some Role",
        "jobSummary": "A short summary.",
        "minimumQualifications": "Some quals.",
    }
    html = _apple_html(minimal)
    result = extract_detail(html)
    assert result is not None
    assert "A short summary." in result.raw_jd_text
    assert result.locations is None
    assert result.posted_date is None


def test_apple_hydration_not_matched_by_unrelated_json_parse_call():
    # A page can legitimately have OTHER JSON.parse(...) calls (analytics
    # config, feature flags) -- only one with the job-shaped keys counts.
    html = (
        '<script>var x = JSON.parse("{\\"unrelated\\": \\"config data\\"}");</script>'
        + _apple_html(APPLE_JOB)
    )
    result = extract_detail(html)
    assert result is not None
    assert "world-class Thermal Engineer" in result.raw_jd_text


# ---- schema.org JobPosting JSON-LD ----

JSON_LD_HTML = """
<html><head>
<script type="application/ld+json">
{
  "@context": "https://schema.org/",
  "@type": "JobPosting",
  "title": "Senior Backend Engineer",
  "description": "<p>Build scalable backend systems for our platform.</p>",
  "datePosted": "2026-08-15",
  "jobLocation": {
    "@type": "Place",
    "address": {
      "@type": "PostalAddress",
      "addressLocality": "Austin",
      "addressRegion": "TX"
    }
  },
  "baseSalary": {
    "@type": "MonetaryAmount",
    "currency": "USD",
    "value": {
      "@type": "QuantitativeValue",
      "minValue": 120000,
      "maxValue": 160000,
      "unitText": "YEAR"
    }
  }
}
</script>
</head><body></body></html>
"""


def test_json_ld_extracts_description_date_and_location():
    result = extract_detail(JSON_LD_HTML)
    assert result is not None
    assert "Build scalable backend systems" in result.raw_jd_text
    assert result.posted_date.isoformat() == "2026-08-15"
    assert result.locations == ["Austin, TX"]


def test_json_ld_base_salary_folded_into_raw_jd_text():
    result = extract_detail(JSON_LD_HTML)
    assert "$120000" in result.raw_jd_text
    assert "$160000" in result.raw_jd_text


def test_json_ld_ignored_when_type_is_not_job_posting():
    html = """
    <html><head>
    <script type="application/ld+json">
    {"@type": "Organization", "name": "Some Company"}
    </script>
    </head><body><p>plain body text that is definitely long enough to pass the two hundred character minimum required for the visible-text fallback tier to actually accept it as real job content rather than rejecting it as too short</p></body></html>
    """
    result = extract_detail(html)
    # Falls through to tier 3 (visible text), not tier 1.
    assert result is not None
    assert "plain body text" in result.raw_jd_text


def test_json_ld_takes_priority_over_apple_hydration_when_both_present():
    html = JSON_LD_HTML.replace("</body>", _apple_html(APPLE_JOB) + "</body>")
    result = extract_detail(html)
    assert "Build scalable backend systems" in result.raw_jd_text
    assert "world-class Thermal Engineer" not in result.raw_jd_text


# ---- Visible-text fallback ----


def test_visible_text_fallback_when_no_structured_data_present():
    html = (
        "<html><head><style>.x{color:red}</style><script>var x = 1;</script></head>"
        "<body><nav>Home About</nav>"
        "<h1>Machine Learning Engineer</h1>"
        "<p>About the Role: We build ranking systems for millions of users worldwide "
        "every single day of the year across many different products and services.</p>"
        "<p>Responsibilities: Design and ship ranking models used in production systems.</p>"
        "</body></html>"
    )
    result = extract_detail(html)
    assert result is not None
    assert "ranking systems" in result.raw_jd_text
    assert "<script>" not in result.raw_jd_text
    assert "color:red" not in result.raw_jd_text
    assert result.posted_date is None
    assert result.locations is None


def test_visible_text_fallback_rejects_too_short_pages():
    html = "<html><body><p>404 Not Found</p></body></html>"
    result = extract_detail(html)
    assert result is None


def test_extract_detail_none_when_every_tier_fails():
    html = "<html><body></body></html>"
    assert extract_detail(html) is None


# ---- fetch_and_extract_detail (fetch + extract combo) ----


class _FakeFetch(FetchStrategy):
    def __init__(self, html: str):
        self.html = html
        self.calls: list[str] = []

    def fetch(self, url: str) -> str:
        self.calls.append(url)
        return self.html


def test_fetch_and_extract_detail_uses_the_given_fetch_strategy():
    fetch = _FakeFetch(_apple_html(APPLE_JOB))
    result = fetch_and_extract_detail("https://jobs.apple.com/details/123", fetch_strategy=fetch)
    assert result is not None
    assert fetch.calls == ["https://jobs.apple.com/details/123"]
