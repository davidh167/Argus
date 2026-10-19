"""Detail-page hydration for scraped postings: a scraped
listing page only ever yields a short blurb (Apple's average is 571 chars
vs 6-14k for crawled sources), not the full JD -- no quals, no structured
location, no pay. This fetches each posting's own URL and extracts full
text (+ location, + posted date) from whichever extraction tier succeeds
first, cheapest/most-generic first:

  1. schema.org JobPosting JSON-LD -- a documented, site-agnostic standard.
     No real site in this project currently emits it (Apple doesn't), so
     this tier is built against the public schema.org spec, not a live
     sample -- narrower than tier 2 as a result (description/datePosted/
     jobLocation/baseSalary only, no attempt at every optional field).
  2. Apple's embedded hydration JSON -- calibrated against live fetches of
     real Apple job pages (see tests/test_detail_fetch.py for exactly
     which fields). The job data sits inside a JS statement
     (`window.__staticRouterHydrationData = JSON.parse("...")`), not a
     `<script type="application/json">` block, so it has to be located and
     JS-string-unescaped before it's valid JSON -- then found by key
     signature (postingTitle/jobSummary/minimumQualifications) rather than
     a hardcoded object path, so a nesting change on Apple's side alone
     doesn't break this.
  3. Visible text (scripts/styles stripped) -- last resort for any site
     that defeats both of the above.

Compensation is deliberately NOT extracted here as a separate field: every
tier folds whatever pay text it finds into raw_jd_text (Apple's real "Pay
& Benefits" content is prose, not a structured field, so there's nothing
more structured to extract), and upsert_posting's existing JD-text
extractor picks it up from there -- the same one mechanism every other
source already goes through, not a second comp path calibrated on guesses.
"""

import json
import re
from dataclasses import dataclass
from datetime import date, datetime

from app.crawlers.base import clean_html_text
from app.crawlers.fetch import FetchStrategy, HttpxFetch


@dataclass
class DetailExtraction:
    raw_jd_text: str
    posted_date: date | None
    locations: list[str] | None


def _compose_raw_jd_text(
    *,
    summary: str | None = None,
    description: str | None = None,
    responsibilities: str | None = None,
    minimum_qualifications: str | None = None,
    preferred_qualifications: str | None = None,
    trailing: str | None = None,
) -> str:
    """Explicit section headers matching jd_sections.py's own keyword
    lists (SUMMARY_HEADERS/"about the role", RESPONSIBILITIES_HEADERS/
    "responsibilities", REQUIRED_HEADERS/"minimum qualifications",
    PREFERRED_HEADERS/"preferred qualifications") so extract_jd_sections
    finds them reliably, regardless of whether the source text happens to
    contain recognizable headers of its own. summary and description are
    folded into one "About the Role" block -- schema.org JobPosting only
    ever gives one prose description with no further breakdown, and
    Apple's jobSummary/description are both scene-setting prose too.
    trailing (e.g. Apple's Pay & Benefits text) is appended after every
    recognized section, letting it fall inside whichever section
    (usually preferred quals or nothing at all) precedes the end of text --
    the generic comp extractor scans the whole raw_jd_text regardless of
    which section it lands in.
    """
    parts = []
    lead = "\n\n".join(p for p in (summary, description) if p)
    if lead:
        parts.append(f"About the Role\n{lead}")
    if responsibilities:
        parts.append(f"Responsibilities\n{responsibilities}")
    if minimum_qualifications:
        parts.append(f"Minimum Qualifications\n{minimum_qualifications}")
    if preferred_qualifications:
        parts.append(f"Preferred Qualifications\n{preferred_qualifications}")
    if trailing:
        parts.append(trailing)
    return "\n\n".join(parts)


# ---- Tier 1: schema.org JobPosting JSON-LD ----

_LD_JSON_RE = re.compile(
    r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.S | re.I
)


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _json_ld_locations(job_location) -> list[str] | None:
    entries = job_location if isinstance(job_location, list) else [job_location]
    locations = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        address = entry.get("address")
        if isinstance(address, dict):
            city = address.get("addressLocality")
            region = address.get("addressRegion")
            place = ", ".join(p for p in (city, region) if p)
            if place:
                locations.append(place)
    return locations or None


def _json_ld_comp_text(base_salary) -> str | None:
    if not isinstance(base_salary, dict):
        return None
    value = base_salary.get("value")
    if not isinstance(value, dict):
        return None
    min_value, max_value = value.get("minValue"), value.get("maxValue")
    if min_value is None or max_value is None:
        return None
    currency = base_salary.get("currency") or ""
    unit = (value.get("unitText") or "").upper()
    period = "per hour" if unit == "HOUR" else "per year"
    return f"Compensation: ${min_value} - ${max_value} {currency} {period}".strip()


def _extract_json_ld(html: str) -> DetailExtraction | None:
    for match in _LD_JSON_RE.finditer(html):
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        candidates = data if isinstance(data, list) else [data]
        for candidate in candidates:
            if not isinstance(candidate, dict) or candidate.get("@type") != "JobPosting":
                continue
            description = clean_html_text(candidate.get("description"))
            if not description:
                continue
            text = _compose_raw_jd_text(
                description=description,
                trailing=_json_ld_comp_text(candidate.get("baseSalary")),
            )
            return DetailExtraction(
                raw_jd_text=text,
                posted_date=_parse_iso_date(candidate.get("datePosted")),
                locations=_json_ld_locations(candidate.get("jobLocation")),
            )
    return None


# ---- Tier 2: Apple's embedded hydration JSON ----

# A proper JS string-literal body: any escaped char, or any char that's
# neither a bare quote nor a bare backslash -- handles escaped quotes
# inside the JSON payload without the match ending prematurely.
_JSON_PARSE_RE = re.compile(r'JSON\.parse\("((?:\\.|[^"\\])*)"\)')

# The exact field names confirmed on a live Apple job page -- see this
# module's docstring. Matched by presence, not by the exact object path
# (loaderData.jobDetails.jobsData today), so a nesting change alone
# doesn't silently stop this from finding the data.
_APPLE_JOB_KEYS = {"postingTitle", "jobSummary", "minimumQualifications"}


def _find_apple_job_data(obj) -> dict | None:
    if isinstance(obj, dict):
        if _APPLE_JOB_KEYS.issubset(obj.keys()):
            return obj
        for value in obj.values():
            found = _find_apple_job_data(value)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_apple_job_data(item)
            if found:
                return found
    return None


def _apple_locations(locations) -> list[str] | None:
    if not isinstance(locations, list):
        return None
    result = []
    for loc in locations:
        if not isinstance(loc, dict):
            continue
        city = loc.get("city")
        region = loc.get("stateProvince") or loc.get("countryName")
        place = ", ".join(p for p in (city, region) if p)
        if place:
            result.append(place)
    return result or None


def _apple_comp_text(posting_footers) -> str | None:
    """postingFooters[].localizations.en_US[] items are named sections
    ("Pay & Benefits", others); only the pay one has dollar text, and it's
    prose HTML, not a structured field -- cleaned and appended so the
    generic JD-text comp extractor (run uniformly by upsert_posting) finds
    it, same as every other source."""
    if not isinstance(posting_footers, list):
        return None
    for footer in posting_footers:
        if not isinstance(footer, dict):
            continue
        for section in (footer.get("localizations") or {}).get("en_US") or []:
            if not isinstance(section, dict):
                continue
            content = section.get("content")
            if content and "pay" in (section.get("name") or "").lower():
                cleaned = clean_html_text(content)
                if cleaned:
                    return cleaned
    return None


def _extract_apple_hydration(html: str) -> DetailExtraction | None:
    for match in _JSON_PARSE_RE.finditer(html):
        try:
            json_text = json.loads('"' + match.group(1) + '"')
            data = json.loads(json_text)
        except (json.JSONDecodeError, ValueError):
            continue
        job = _find_apple_job_data(data)
        if not job:
            continue

        text = _compose_raw_jd_text(
            summary=clean_html_text(job.get("jobSummary")),
            description=clean_html_text(job.get("description")),
            responsibilities=clean_html_text(job.get("responsibilities")),
            minimum_qualifications=clean_html_text(job.get("minimumQualifications")),
            preferred_qualifications=clean_html_text(job.get("preferredQualifications")),
            trailing=_apple_comp_text(job.get("postingFooters")),
        )
        if not text:
            continue

        posted = None
        post_date_gmt = job.get("postDateInGMT")
        if post_date_gmt:
            try:
                posted = datetime.fromisoformat(post_date_gmt.replace("Z", "+00:00")).date()
            except ValueError:
                pass

        return DetailExtraction(
            raw_jd_text=text,
            posted_date=posted,
            locations=_apple_locations(job.get("locations")),
        )
    return None


# ---- Tier 3: visible text, last resort ----

_SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.S | re.I)


def _extract_visible_text(html: str) -> DetailExtraction | None:
    stripped = _SCRIPT_STYLE_RE.sub(" ", html)
    text = clean_html_text(stripped)
    if not text or len(text) < 200:  # too little to be a real JD -- not worth keeping
        return None
    return DetailExtraction(
        raw_jd_text=_compose_raw_jd_text(description=text), posted_date=None, locations=None
    )


def extract_detail(html: str) -> DetailExtraction | None:
    """Tries each tier in order, returning the first success. None only
    when every tier fails outright (e.g. a fetch that returned an error
    page) -- the caller should treat that as "hydration failed for this
    posting," not raise."""
    for tier in (_extract_json_ld, _extract_apple_hydration, _extract_visible_text):
        result = tier(html)
        if result:
            return result
    return None


def fetch_and_extract_detail(
    url: str, fetch_strategy: FetchStrategy | None = None
) -> DetailExtraction | None:
    fetch_strategy = fetch_strategy or HttpxFetch()
    html = fetch_strategy.fetch(url)
    return extract_detail(html)
