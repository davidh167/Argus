from datetime import datetime

from sqlalchemy import Boolean, DateTime, Index, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str | None] = mapped_column(Text)
    ats_type: Mapped[str | None] = mapped_column(Text)  # greenhouse|lever|ashby|workday|custom
    careers_url: Mapped[str | None] = mapped_column(Text)
    last_crawled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # The careers_url an authoritative crawl actually ran against, recorded
    # so the next crawl can tell whether careers_url changed since then --
    # if it did, that crawl covers a different set of postings, and closure
    # marking (see companies.crawl_company / scraper_catalog.scrape_company)
    # must be skipped for that one run rather than wrongly closing postings
    # that simply fell outside the new URL's scope.
    last_crawl_url: Mapped[str | None] = mapped_column(Text)
    crawl_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[str | None] = mapped_column(Text)

    postings: Mapped[list["Posting"]] = relationship(back_populates="company")

    __table_args__ = (
        Index(
            "uq_companies_domain",
            "domain",
            unique=True,
            postgresql_where=text("domain IS NOT NULL"),
        ),
    )
