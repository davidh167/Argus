"""Cleanup for postings captured from linkedin.com/jobs pages.

The browser extension captures document.body.innerText, which on LinkedIn is
the whole page: nav chrome, the job header, the description, a company
blurb, a sidebar of OTHER recommended jobs, a footer, and ~40 language
names. Confirmed live against a real capture (28.7K chars, of which the
actual description was ~11.6K): feeding that to Stage 2/3 means the model
sees unrelated postings' titles/salaries as part of this one.

LinkedIn's DOM class names are unstable and can't be verified without a
logged-in session, so this cleans the captured TEXT instead, anchored on the
section headings LinkedIn renders ("About the job", "About the company")
rather than on markup. Anything that doesn't look like a LinkedIn job page
passes through untouched.
"""

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

_JOB_ID = re.compile(r"/jobs/view/(?:[^/]*-)?(\d+)")
# Search-results and recommended-list pages keep the selected job's id in
# the query string instead of the path (/jobs/search-results/?currentJobId=
# 123, /jobs/collections/recommended/?currentJobId=123); the right-hand pane
# on those pages shows the same "About the job" layout as a single-job page.
_HEADER_WINDOW = 2000  # chars of header nearest "About the job" to search for location/workplace
# "Mountain View, CA · 1 week ago · Over 100 applicants" -- the "N units ago"
# segment is what distinguishes this from neighboring lines that also use
# " · " (e.g. "Promoted by hirer · Actively reviewing applicants").
_LOCATION_LINE = re.compile(r"^(.+?) · (?:Reposted )?\d+\s+\w+\s+ago\b", re.I)
_WORKPLACE_LINES = {"remote": "remote", "hybrid": "hybrid", "on-site": "onsite", "onsite": "onsite"}
# End-of-description markers, earliest match wins. "About the company" comes
# first on a normal page; the rest cover pages that omit that section.
_END_MARKERS = ("About the company", "More jobs", "See more jobs", "Looking for talent")


@dataclass
class LinkedInCapture:
    url: str
    text: str
    location: str | None
    workplace_type: str | None


def _job_id(url: str) -> str | None:
    parts = urlsplit(url)
    if parts.netloc.lower().removeprefix("www.") != "linkedin.com":
        return None
    match = _JOB_ID.search(parts.path)
    if match:
        return match.group(1)
    if parts.path.startswith("/jobs/"):
        ids = parse_qs(parts.query).get("currentJobId")
        if ids and ids[0].isdigit():
            return ids[0]
    return None


def is_linkedin_job_url(url: str | None) -> bool:
    return bool(url) and _job_id(url) is not None


def canonical_url(url: str) -> str:
    """Drops tracking query params (trk, refId, trackingId, ...) so the same
    job captured via two different links dedupes to one posting."""
    return f"https://www.linkedin.com/jobs/view/{_job_id(url)}/"


def clean_capture(url: str, text: str) -> LinkedInCapture:
    if not is_linkedin_job_url(url):
        return LinkedInCapture(url=url, text=text, location=None, workplace_type=None)

    start = text.find("About the job")
    if start == -1:
        # Not the layout we know how to cut -- keep the text, still fix the URL.
        return LinkedInCapture(
            url=canonical_url(url), text=text, location=None, workplace_type=None
        )

    header = text[:start]
    body = text[start:]
    ends = [i for i in (body.find(m) for m in _END_MARKERS) if i > 0]
    if ends:
        body = body[: min(ends)]

    return LinkedInCapture(
        url=canonical_url(url),
        text=body.strip(),
        location=_location_from_header(header),
        workplace_type=_workplace_from_header(header),
    )


def _location_from_header(header: str) -> str | None:
    # The line right after the title reads "Mountain View, CA · 1 hour ago ·
    # Over 100 applicants" -- the location is whatever precedes the first dot.
    # Reversed and windowed: on a list page the header also holds every
    # other job card, and the selected job's own line is the nearest one.
    for line in reversed(header[-_HEADER_WINDOW:].splitlines()):
        line = line.strip()
        match = _LOCATION_LINE.match(line)
        if match:
            return match.group(1).strip()
    return None


def _workplace_from_header(header: str) -> str | None:
    for line in reversed(header[-_HEADER_WINDOW:].splitlines()):
        value = _WORKPLACE_LINES.get(line.strip().lower())
        if value:
            return value
    return None
