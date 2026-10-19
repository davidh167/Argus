import html
import re
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlsplit


@dataclass
class NormalizedJob:
    external_id: str
    title: str
    url: str | None
    raw_jd_text: str | None
    posted_date: date | None
    locations: list[str] | None = None
    workplace_type: str | None = None  # onsite|hybrid|remote
    department: str | None = None
    # Only ever set from an ATS-structured field (Lever's salaryRange) --
    # comp pulled from JD text is extracted downstream in
    # postings.upsert_posting, uniformly across every source, not per-crawler.
    comp_min: int | None = None
    comp_max: int | None = None
    comp_currency: str | None = None
    comp_period: str | None = None  # year|hour


def infer_workplace_type_from_locations(locations: list[str] | None) -> str | None:
    """Best-effort fallback for a job with no explicit workplace-type field:
    if every location string mentions "remote", call it remote. Anything
    else is genuinely unknown -- "on-site" can't be inferred from a city
    name alone (an on-site posting's location is indistinguishable from a
    hybrid one's), so this deliberately only ever returns "remote" or None.
    """
    if not locations:
        return None
    if all("remote" in loc.lower() for loc in locations):
        return "remote"
    return None


# Matches the closing/self-closing tags that mark a real paragraph or list-item
# boundary -- the ones worth turning into a newline before the rest of the
# markup is stripped to nothing. Deliberately not every block tag (no
# <table>/<tr> handling, no source has needed it yet).
_BLOCK_BOUNDARY_RE = re.compile(r"</(?:p|li|div|h[1-6])\s*>|<br\s*/?>", re.IGNORECASE)
_ANY_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RUN_RE = re.compile(r"[ \t]+")
_NEWLINE_RUN_RE = re.compile(r"\s*\n\s*")


def clean_html_text(raw: str | None) -> str | None:
    """Turns Greenhouse's `content` field -- HTML markup that is itself
    HTML-escaped, e.g. `&lt;div class=&quot;x&quot;&gt;` encoding a literal
    `<div class="x">` -- into plain, paragraph-broken text.

    Order matters: unescape FIRST to recover the real tags, THEN convert
    block boundaries to newlines, THEN strip whatever tags remain. Stripping
    tags first (the old _strip_html behavior) does nothing to escaped
    markup -- there's no literal `<...>` for a `<[^>]+>` regex to match --
    which is exactly why every Greenhouse posting's text was full of
    `&lt;div class=&quot;...` noise before this existed.

    Safe to call on already-plain text (no entities, no tags): unescape and
    both tag regexes are no-ops on it, so this also works as Apple detail
    hydration's shared text cleaner.

    Unescapes TWICE: Greenhouse's content is escaped at two levels, not
    uniformly -- the tag structure (`&lt;div&gt;`) is single-escaped, but
    inline entities the original HTML author used (`&amp;mdash;` for an
    em-dash, `&amp;nbsp;` for a non-breaking space) are double-escaped, so
    one unescape pass leaves literal `&mdash;`/`&nbsp;` text sitting in the
    output instead of "-"/" " -- confirmed against Anthropic's real "Annual
    Salary: $270,000 &amp;mdash; $340,000 USD" widget, where a single pass
    left the literal string "&mdash;" between the two figures, breaking any
    attempt to parse the range (see app/services/compensation.py). A second
    unescape pass is a safe no-op on text that was only single-escaped
    (`html.unescape` doesn't touch a string with no entity references left).
    """
    if not raw:
        return raw
    text = html.unescape(html.unescape(raw))
    text = _BLOCK_BOUNDARY_RE.sub("\n", text)
    text = _ANY_TAG_RE.sub(" ", text)
    text = _SPACE_RUN_RE.sub(" ", text)
    text = _NEWLINE_RUN_RE.sub("\n", text)
    return text.strip()


def extract_slug(careers_url: str, ats_type: str) -> str | None:
    """Best-effort ATS board slug extraction from a careers_url, for the
    known Greenhouse/Lever/Ashby URL shapes (e.g. boards.greenhouse.io/
    {slug}, jobs.lever.co/{slug}, jobs.ashbyhq.com/{slug} -- confirmed
    against a real board URL)."""
    parts = urlsplit(careers_url)
    path_parts = [p for p in parts.path.split("/") if p]
    if not path_parts:
        return None
    if ats_type == "greenhouse" and "greenhouse.io" in parts.netloc:
        return path_parts[0]
    if ats_type == "lever" and "lever.co" in parts.netloc:
        return path_parts[0]
    if ats_type == "ashby" and "ashbyhq.com" in parts.netloc:
        return path_parts[0]
    return None
