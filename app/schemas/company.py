from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class CompanyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    domain: str | None
    ats_type: str | None
    careers_url: str | None
    last_crawled_at: datetime | None
    crawl_enabled: bool
    notes: str | None
    # Computed at read time from postings, not stored (see list_companies) --
    # 0/0 for a brand-new company with nothing crawled yet.
    total_postings: int = 0
    open_postings: int = 0


class CompanyCreate(BaseModel):
    name: str
    careers_url: str | None = None
    ats_type: Literal["greenhouse", "lever", "ashby"] | None = None
    notes: str | None = None


class CrawlResult(BaseModel):
    jobs_found: int
    created: int
    updated: int
    closed: int
    reopened: int
    # False when closure marking was skipped this run (zero jobs returned,
    # or careers_url changed since the last crawl -- see
    # companies.crawl_company) -- distinguishes "0 postings closed because
    # none actually closed" from "closure wasn't even evaluated," which a
    # bare closed=0 can't.
    closure_evaluated: bool


class ScrapeResult(BaseModel):
    jobs_found: int
    created: int
    updated: int
    closed: int
    reopened: int
    fallback_used: bool
    closure_evaluated: bool
    # How many of this scrape's newly created postings got their detail
    # page fetched inline (see scraper_catalog.SCRAPE_INLINE_HYDRATE_CAP) --
    # the rest of a large backlog is left for POST /postings/hydrate.
    hydrated_inline: int


class RefreshAllEntry(BaseModel):
    company_id: int
    company_name: str
    ok: bool
    # A loose dict rather than CrawlResult | ScrapeResult -- a crawl-type
    # and scrape-type company can both appear in the same refresh-all run,
    # and each already has its own typed shape at its own endpoint.
    result: dict | None = None
    error: str | None = None


class RefreshAllResult(BaseModel):
    results: list[RefreshAllEntry]
