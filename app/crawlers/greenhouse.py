from datetime import date

import httpx

from app.crawlers.base import NormalizedJob, clean_html_text, infer_workplace_type_from_locations

GREENHOUSE_API = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"

# Every value observed on a real board (Anthropic's): "On-Site", "Remote",
# "Hybrid (Travel-Required)". Matched by prefix so a variant like "Hybrid"
# alone (no travel qualifier) still maps correctly.
_LOCATION_TYPE_PREFIXES = (
    ("on-site", "onsite"),
    ("onsite", "onsite"),
    ("hybrid", "hybrid"),
    ("remote", "remote"),
)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _workplace_type(job: dict, locations: list[str]) -> str | None:
    for metadatum in job.get("metadata") or []:
        if metadatum.get("name") == "Location Type":
            value = (metadatum.get("value") or "").lower()
            for prefix, mapped in _LOCATION_TYPE_PREFIXES:
                if value.startswith(prefix):
                    return mapped
            return None
    # No "Location Type" metadata at all (~9% of Anthropic's live board) --
    # fall back to inferring remote from the location text itself.
    return infer_workplace_type_from_locations(locations)


def fetch_greenhouse_jobs(slug: str) -> list[NormalizedJob]:
    resp = httpx.get(GREENHOUSE_API.format(slug=slug), params={"content": "true"}, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    jobs = []
    for j in data.get("jobs", []):
        # first_published is when the listing actually went live; updated_at
        # moves on any edit (a typo fix, a reworded bullet), which made
        # "posted" really mean "last edited." Fall back to updated_at only
        # when first_published is absent (not observed in real data, but
        # not every ATS-compatible board is guaranteed to send it).
        posted = _parse_date(j.get("first_published")) or _parse_date(j.get("updated_at"))

        # location.name is a single "|"-joined display string covering
        # every office/region the posting lists, e.g. "San Francisco, CA |
        # New York City, NY" -- confirmed against Anthropic's live board.
        location_name = (j.get("location") or {}).get("name") or ""
        locations = [part.strip() for part in location_name.split("|") if part.strip()]

        departments = j.get("departments") or []
        department = departments[0].get("name") if departments else None

        jobs.append(
            NormalizedJob(
                external_id=str(j["id"]),
                title=j.get("title", ""),
                url=j.get("absolute_url"),
                raw_jd_text=clean_html_text(j.get("content")),
                posted_date=posted,
                locations=locations or None,
                workplace_type=_workplace_type(j, locations),
                department=department,
            )
        )
    return jobs
