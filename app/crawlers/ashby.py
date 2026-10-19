from datetime import date, datetime

import httpx

from app.crawlers.base import NormalizedJob

ASHBY_API = "https://api.ashbyhq.com/posting-api/job-board/{slug}"


def _comp_period(interval: str | None) -> str | None:
    """Ashby's real interval value ("1 YEAR") isn't one of our year|hour
    convention values -- normalize by substring match, same reasoning as
    lever._comp_period (not shared verbatim: the two APIs' exact interval
    spellings differ enough that a shared helper would need the same
    broad substring check anyway)."""
    if not interval:
        return None
    interval_lower = interval.lower()
    if "year" in interval_lower:
        return "year"
    if "hour" in interval_lower:
        return "hour"
    return None


def fetch_ashby_jobs(slug: str) -> list[NormalizedJob]:
    """Ashby's public job-board API -- checked against several
    real boards. descriptionPlain is already
    clean plain text (no HTML stripping needed, unlike Greenhouse).
    workplaceType lowercases directly to this schema's onsite|hybrid|
    remote convention. Every job seen across all 4 boards had
    isListed=true -- the API appears to only ever return live listings,
    so closure detection works the same way as Greenhouse/Lever
    (not-seen-this-crawl = closed)."""
    resp = httpx.get(
        ASHBY_API.format(slug=slug), params={"includeCompensation": "true"}, timeout=30
    )
    resp.raise_for_status()
    data = resp.json()

    jobs = []
    for j in data.get("jobs", []):
        posted: date | None = None
        published_at = j.get("publishedAt")
        if published_at:
            try:
                posted = datetime.fromisoformat(published_at).date()
            except ValueError:
                pass

        locations = [j["location"]] if j.get("location") else []
        for secondary in j.get("secondaryLocations") or []:
            if secondary.get("location"):
                locations.append(secondary["location"])

        workplace_type = (j.get("workplaceType") or "").lower() or None

        comp_min = comp_max = comp_currency = comp_period = None
        compensation = j.get("compensation") or {}
        for component in compensation.get("summaryComponents") or []:
            if component.get("compensationType") != "Salary":
                continue
            period = _comp_period(component.get("interval"))
            if (
                period
                and isinstance(component.get("minValue"), (int, float))
                and isinstance(component.get("maxValue"), (int, float))
            ):
                comp_min = int(component["minValue"])
                comp_max = int(component["maxValue"])
                comp_currency = (component.get("currencyCode") or "USD").upper()
                comp_period = period
                break

        jobs.append(
            NormalizedJob(
                external_id=str(j.get("id", "")),
                title=j.get("title", ""),
                url=j.get("jobUrl"),
                raw_jd_text=j.get("descriptionPlain") or "",
                posted_date=posted,
                locations=locations or None,
                workplace_type=workplace_type,
                department=j.get("department"),
                comp_min=comp_min,
                comp_max=comp_max,
                comp_currency=comp_currency,
                comp_period=comp_period,
            )
        )
    return jobs
