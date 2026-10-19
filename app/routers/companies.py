from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.company import Company
from app.models.posting import Posting
from app.schemas.company import (
    CompanyCreate,
    CompanyOut,
    CrawlResult,
    RefreshAllResult,
    ScrapeResult,
)
from app.services import relevance as relevance_service
from app.services.companies import (
    UnsupportedAtsError,
    crawl_company,
    extract_domain,
    refresh_all_companies,
)
from app.services.scraper_catalog import (
    ScraperNotApplicableError,
    ScraperValidationFailedError,
    scrape_company,
)
from app.services.stage2 import Stage2UnavailableError

router = APIRouter(prefix="/companies", tags=["companies"])


def _to_company_out(company: Company, total: int = 0, open_count: int = 0) -> CompanyOut:
    return CompanyOut(
        id=company.id,
        name=company.name,
        domain=company.domain,
        ats_type=company.ats_type,
        careers_url=company.careers_url,
        last_crawled_at=company.last_crawled_at,
        crawl_enabled=company.crawl_enabled,
        notes=company.notes,
        total_postings=total,
        open_postings=open_count,
    )


@router.post("", response_model=CompanyOut)
def create_company(payload: CompanyCreate, db: Session = Depends(get_db)):
    company = Company(
        name=payload.name,
        careers_url=payload.careers_url,
        domain=extract_domain(payload.careers_url),
        ats_type=payload.ats_type,
        notes=payload.notes,
    )
    db.add(company)
    db.commit()
    db.refresh(company)
    return _to_company_out(company)


@router.get("", response_model=list[CompanyOut])
def list_companies(db: Session = Depends(get_db)):
    companies = list(db.scalars(select(Company).order_by(Company.name)))
    # One grouped query for every company's counts, not N+1 per-company
    # queries.
    counts = db.execute(
        select(
            Posting.company_id,
            func.count(Posting.id),
            func.count(Posting.id).filter(Posting.closed_at.is_(None)),
        ).group_by(Posting.company_id)
    ).all()
    counts_by_company = {
        company_id: (total, open_count) for company_id, total, open_count in counts
    }
    return [_to_company_out(c, *counts_by_company.get(c.id, (0, 0))) for c in companies]


@router.post("/refresh-all", response_model=RefreshAllResult)
def refresh_all(db: Session = Depends(get_db)):
    """Crawls or scrapes every enabled, configured company sequentially
    (5.2) -- the same per-company Crawl/Scrape action the Companies tab's
    individual buttons already trigger, just looped. A company with no
    careers_url is silently skipped (never a real crawl target); a
    per-company failure is recorded in its own entry rather than aborting
    the run. rescore_all runs once at the end, not per company."""
    results = refresh_all_companies(db)
    relevance_service.rescore_all(db)
    db.commit()
    return RefreshAllResult(results=results)


@router.post("/{company_id}/crawl", response_model=CrawlResult)
def crawl(company_id: int, db: Session = Depends(get_db)):
    company = db.get(Company, company_id)
    if company is None:
        raise HTTPException(status_code=404, detail="Company not found")
    try:
        result = crawl_company(db, company)
    except UnsupportedAtsError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    # New/changed text is worth reflecting in relevance -- cheap (measured
    # ~0.85s over the whole real 2,744-posting corpus).
    relevance_service.rescore_all(db)
    db.commit()
    return result


@router.post("/{company_id}/scrape", response_model=ScrapeResult)
def scrape(company_id: int, db: Session = Depends(get_db)):
    company = db.get(Company, company_id)
    if company is None:
        raise HTTPException(status_code=404, detail="Company not found")
    try:
        result = scrape_company(db, company)
    except ScraperNotApplicableError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Stage2UnavailableError:
        raise HTTPException(
            status_code=400,
            detail="ANTHROPIC_API_KEY not configured; required to generate a new site scraper strategy.",
        ) from None
    except ScraperValidationFailedError as e:
        raise HTTPException(status_code=502, detail=str(e)) from None
    relevance_service.rescore_all(db)
    db.commit()
    return result
