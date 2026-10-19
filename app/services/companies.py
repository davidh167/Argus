from datetime import datetime, timezone
from urllib.parse import urlsplit

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.crawlers.ashby import fetch_ashby_jobs
from app.crawlers.base import extract_slug
from app.crawlers.greenhouse import fetch_greenhouse_jobs
from app.crawlers.lever import fetch_lever_jobs
from app.models.company import Company


class UnsupportedAtsError(RuntimeError):
    pass


# Multi-tenant ATS hosts: many unrelated companies share this exact
# hostname, so the hostname alone is not a valid company identity --
# the real per-company distinction lives in the URL's first path segment
# (the board slug, e.g. "/acme/", "/globex/"). Keying on the hostname
# alone silently merged captures from different companies into whichever
# company had first claimed the bare hostname as its domain.
_MULTI_TENANT_ATS_HOSTS = {
    "job-boards.greenhouse.io",
    "boards.greenhouse.io",
    "jobs.lever.co",
    "jobs.ashbyhq.com",
}


def extract_domain(url: str | None) -> str | None:
    if not url:
        return None
    parts = urlsplit(url if "://" in url else f"//{url}")
    netloc = parts.netloc.lower().removeprefix("www.")
    if not netloc:
        return None
    if netloc in _MULTI_TENANT_ATS_HOSTS:
        path_parts = [p for p in parts.path.split("/") if p]
        if path_parts:
            return f"{netloc}/{path_parts[0]}"
        return None  # a bare multi-tenant host with no slug identifies no single company
    return netloc


def get_or_create_company(db: Session, name: str, url: str | None = None) -> Company:
    """Resolve a company by domain first (more reliable), then case-insensitive
    name match, else create it. Domain has a partial unique index, so this is
    race-safe even if it's never a real concern for a single-user local tool."""
    domain = extract_domain(url)

    if domain:
        existing = db.scalar(select(Company).where(Company.domain == domain))
        if existing:
            return existing

    existing = db.scalar(select(Company).where(func.lower(Company.name) == name.lower()))
    if existing:
        if domain and not existing.domain:
            existing.domain = domain
        return existing

    company = Company(name=name, domain=domain)
    db.add(company)
    db.flush()  # assign company.id without committing the outer transaction
    return company


def crawl_company(db: Session, company: Company) -> dict:
    """Fetch all listings from a company's known ATS API and upsert them as
    postings, running Stage 1 on each. Covers the ATS APIs (Greenhouse, Lever, Ashby);
    everything else goes through the scraper catalog (scrape_company).

    Local import to avoid a companies<->postings circular import: postings.py
    already imports get_or_create_company from this module.
    """
    from app.services.postings import mark_closed_postings, upsert_posting, write_stage1_verdict

    if not company.ats_type or not company.careers_url:
        raise UnsupportedAtsError(
            "Company needs ats_type ('greenhouse'|'lever'|'ashby') and careers_url set before crawling."
        )

    slug = extract_slug(company.careers_url, company.ats_type)
    if not slug:
        raise UnsupportedAtsError(
            f"Could not extract a board slug from careers_url={company.careers_url!r}"
        )

    if company.ats_type == "greenhouse":
        jobs = fetch_greenhouse_jobs(slug)
        source = "greenhouse_api"
    elif company.ats_type == "lever":
        jobs = fetch_lever_jobs(slug)
        source = "lever_api"
    elif company.ats_type == "ashby":
        jobs = fetch_ashby_jobs(slug)
        source = "ashby_api"
    else:
        raise UnsupportedAtsError(f"Unsupported ats_type: {company.ats_type!r}")

    created = updated = reopened_count = 0
    seen_external_ids: set[str] = set()
    seen_urls: set[str] = set()
    for job in jobs:
        if job.external_id:
            seen_external_ids.add(job.external_id)
        if job.url:
            seen_urls.add(job.url)
        posting, was_created, title_changed, reopened = upsert_posting(
            db,
            company,
            title=job.title,
            url=job.url,
            external_id=job.external_id,
            raw_jd_text=job.raw_jd_text,
            posted_date=job.posted_date,
            source=source,
            locations=job.locations,
            workplace_type=job.workplace_type,
            department=job.department,
            comp_min=job.comp_min,
            comp_max=job.comp_max,
            comp_currency=job.comp_currency,
            comp_period=job.comp_period,
        )
        # Only write a fresh Stage 1 verdict when there's actually a new
        # title to classify -- Stage 1 is title-only (classify_title), so a
        # re-crawl that just refreshes text/dates would otherwise stack an
        # identical verdict on every posting every time, burying any real
        # Stage 2/3 result under it (that verdict list is read latest-first
        # everywhere else in this codebase).
        if was_created or title_changed:
            write_stage1_verdict(db, posting)
        created += int(was_created)
        updated += int(not was_created)
        reopened_count += int(reopened)

    # The Greenhouse/Lever APIs always return the company's full current
    # board -- a non-empty response is inherently a complete listing. Only
    # skip closure when the response was empty (almost certainly a broken
    # fetch, not an actually-empty board -- marking every open posting
    # closed on that basis would be exactly the failure the self-healing
    # scraper catalog's own zero-results guard exists to avoid) or when
    # careers_url changed since the last crawl that set last_crawl_url
    # (that crawl covered a different set of postings, so "not seen" isn't
    # meaningful evidence of closure).
    closed = 0
    url_unchanged = company.last_crawl_url is None or company.last_crawl_url == company.careers_url
    closure_evaluated = bool(jobs) and url_unchanged
    if closure_evaluated:
        closed = mark_closed_postings(db, company, source, seen_external_ids, seen_urls)

    company.last_crawl_url = company.careers_url
    company.last_crawled_at = datetime.now(timezone.utc)
    db.flush()
    return {
        "jobs_found": len(jobs),
        "created": created,
        "updated": updated,
        "closed": closed,
        "reopened": reopened_count,
        "closure_evaluated": closure_evaluated,
    }


def refresh_all_companies(db: Session) -> list[dict]:
    """Crawls or scrapes every enabled, configured company sequentially
    (5.2) -- crawl if ats_type is set, scrape otherwise, the same
    dispatch rule the Companies tab's per-company button already uses.
    Skips a company entirely (not even attempted, not reported as a
    failure) when careers_url is null -- most of the real corpus's
    companies are single manually-captured postings with no crawl
    infrastructure at all, and were never a crawl target to begin with.

    Per-company failures are caught and recorded rather than aborting the
    whole run -- partial failure across independent companies is expected,
    not exceptional, the same reasoning stage2_batch.apply_batch_results
    already uses for per-item batch failures. Local imports to avoid a
    companies<->scraper_catalog circular import risk (scraper_catalog
    doesn't currently import companies, but keeping this consistent with
    crawl_company's own local-import convention in this file)."""
    from app.services.scraper_catalog import (
        ScraperNotApplicableError,
        ScraperValidationFailedError,
        scrape_company,
    )
    from app.services.stage2 import Stage2UnavailableError

    companies = list(
        db.scalars(select(Company).where(Company.crawl_enabled.is_(True)).order_by(Company.name))
    )
    results = []
    for company in companies:
        if not company.careers_url:
            continue
        try:
            result = crawl_company(db, company) if company.ats_type else scrape_company(db, company)
            results.append(
                {
                    "company_id": company.id,
                    "company_name": company.name,
                    "ok": True,
                    "result": result,
                }
            )
        except (
            UnsupportedAtsError,
            ScraperNotApplicableError,
            Stage2UnavailableError,
            ScraperValidationFailedError,
        ) as e:
            results.append(
                {
                    "company_id": company.id,
                    "company_name": company.name,
                    "ok": False,
                    "error": str(e),
                }
            )
    return results
