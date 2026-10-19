from datetime import date, datetime, timezone

import httpx

from app.crawlers.base import NormalizedJob

LEVER_API = "https://api.lever.co/v0/postings/{slug}"


def _comp_period(interval: str | None) -> str | None:
    """Lever's real interval value ("per-year-salary") isn't one of our
    year|hour convention values -- normalize by substring match rather
    than an exact map, and return None (comp unrecognized -- leave it to
    upsert_posting's JD-text fallback) for anything that mentions neither,
    rather than guessing."""
    if not interval:
        return None
    interval_lower = interval.lower()
    if "year" in interval_lower:
        return "year"
    if "hour" in interval_lower:
        return "hour"
    return None


def fetch_lever_jobs(slug: str) -> list[NormalizedJob]:
    resp = httpx.get(LEVER_API.format(slug=slug), params={"mode": "json"}, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    jobs = []
    for j in data:
        posted: date | None = None
        created_at_ms = j.get("createdAt")
        if created_at_ms:
            try:
                posted = datetime.fromtimestamp(created_at_ms / 1000, tz=timezone.utc).date()
            except (ValueError, OSError, OverflowError):
                pass

        description_parts = [j.get("descriptionPlain") or j.get("description") or ""]
        for section in j.get("lists", []) or []:
            if section.get("text"):
                description_parts.append(section["text"])
            content = section.get("content")
            if isinstance(content, str):
                description_parts.append(content)

        categories = j.get("categories") or {}
        locations = [loc for loc in (categories.get("allLocations") or []) if loc]
        if not locations and categories.get("location"):
            locations = [categories["location"]]
        # workplaceType is top-level, not under categories -- confirmed
        # against a real board, and already lowercase onsite|hybrid|remote,
        # matching this schema's convention values directly.
        workplace_type = (j.get("workplaceType") or "").lower() or None
        # department, when present, is more specific than team (e.g.
        # "BevMo!" vs "Leadership" on the same real posting) -- prefer it,
        # falling back to team when a board doesn't set department.
        department = categories.get("department") or categories.get("team")

        salary_range = j.get("salaryRange") or {}
        comp_period = _comp_period(salary_range.get("interval"))
        comp_min = comp_max = comp_currency = None
        if (
            comp_period
            and isinstance(salary_range.get("min"), (int, float))
            and isinstance(salary_range.get("max"), (int, float))
        ):
            comp_min = int(salary_range["min"])
            comp_max = int(salary_range["max"])
            comp_currency = (salary_range.get("currency") or "USD").upper()

        jobs.append(
            NormalizedJob(
                external_id=str(j.get("id", "")),
                title=j.get("text", ""),
                url=j.get("hostedUrl"),
                raw_jd_text="\n".join(p for p in description_parts if p).strip(),
                posted_date=posted,
                locations=locations or None,
                workplace_type=workplace_type,
                department=department,
                comp_min=comp_min,
                comp_max=comp_max,
                comp_currency=comp_currency,
                comp_period=comp_period,
            )
        )
    return jobs
