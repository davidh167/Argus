from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Posting(Base):
    __tablename__ = "postings"

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)

    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str | None] = mapped_column(Text)
    external_id: Mapped[str | None] = mapped_column(Text)
    raw_jd_text: Mapped[str | None] = mapped_column(Text)
    # Boilerplate-free JD text (see jd_sections.build_normalized_jd_text) --
    # relevance scoring and similar-roles matching use
    # this instead of raw_jd_text, which still carries same-company EEO/
    # benefits boilerplate that would otherwise dominate similarity.
    normalized_jd_text: Mapped[str | None] = mapped_column(Text)
    fingerprint: Mapped[str | None] = mapped_column(
        Text
    )  # reserved for near-duplicate detection (not used yet)

    # Display strings as the source gives them (e.g. "San Francisco, CA"),
    # not geocoded -- a posting can list several. JSONB list of str, not
    # Postgres ARRAY, per this schema's established convention (see
    # SkillRoadmap.trend_evidence).
    #
    # none_as_null=True: found live -- a manual capture with no location
    # (capture_posting passes locations=None explicitly) was storing a
    # literal JSON null value, not a true SQL NULL. SQLAlchemy's JSON/JSONB
    # type defaults to none_as_null=False, meaning an explicit Python None
    # assigned to a JSON-typed column is treated as "store a JSON null,"
    # not "leave this NULL" -- unlike every other column type in this
    # schema, where None always means SQL NULL. That silently broke
    # Posting.locations.is_(None) (used by location_match_condition's
    # "unknown location" detection, 2.2) for that row: `'null'::jsonb IS
    # NULL` is false in Postgres, so it read as a definite match instead
    # of "no data." Applied to every JSONB column below, not just this
    # one, since it's a type-level footgun any future caller could
    # rediscover the hard way.
    locations: Mapped[list | None] = mapped_column(JSONB(none_as_null=True))
    workplace_type: Mapped[str | None] = mapped_column(Text)  # onsite|hybrid|remote
    department: Mapped[str | None] = mapped_column(Text)

    # comp_min/comp_max are whole units of comp_currency per comp_period
    # (e.g. 270000 USD per year, or 28 USD per hour) -- never fractional,
    # matching every real figure seen (see app/services/compensation.py).
    # comp_source distinguishes an ATS-structured field (Lever's
    # salaryRange, trusted) from text pulled out of raw_jd_text by
    # compensation.extract_compensation (a best-effort regex match).
    # comp_raw keeps the exact matched snippet(s) so the UI can show what
    # was actually matched, not just the derived numbers.
    comp_min: Mapped[int | None] = mapped_column(Integer)
    comp_max: Mapped[int | None] = mapped_column(Integer)
    comp_currency: Mapped[str | None] = mapped_column(Text)
    comp_period: Mapped[str | None] = mapped_column(Text)  # year|hour
    comp_source: Mapped[str | None] = mapped_column(Text)  # ats_field|jd_text
    comp_raw: Mapped[str | None] = mapped_column(Text)

    # Set once a scraped posting's full detail page has been fetched
    # -- crawled/manually-captured postings already have full
    # text from ingestion, so this stays null for them. Used to exclude
    # unhydrated scraped postings (a short listing blurb) from Stage 2/3
    # batch submission.
    detail_fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # TF-IDF/cosine similarity of normalized_jd_text against the profile's
    # target_signals/skills (app/services/relevance.py), computed for every
    # posting in one pass over the whole corpus by relevance.rescore_all --
    # not computed per-request, since IDF needs the whole corpus. Null
    # until the first rescore (or for a posting that predates it).
    # relevance_profile_version_id records which profile version produced
    # this score, so a stale one (after a profile edit, before the next
    # rescore) is at least identifiable, mirroring stage2/3_profile_version_id.
    relevance_score: Mapped[float | None] = mapped_column(Float)
    relevance_profile_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("profile_versions.id")
    )
    # The ~8 terms contributing most to relevance_score's dot product,
    # highest first -- cheap "why is this ranked here" explainability,
    # computed and stored in the same rescore_all pass rather than
    # recomputed per detail-page request (which would need the whole
    # corpus's IDF again).
    relevance_matched_terms: Mapped[list | None] = mapped_column(JSONB(none_as_null=True))

    posted_date: Mapped[date | None] = mapped_column(Date)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    # Only incremented when a posting that was CLOSED reappears in a later
    # authoritative crawl (a genuine repost) -- not on every re-sighting
    # while still open. See upsert_posting.
    repost_count: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str | None] = mapped_column(
        Text
    )  # manual_capture|greenhouse_api|lever_api|ashby_api|scraped

    # Set when an authoritative crawl of this company+source no longer
    # lists the posting (see companies.crawl_company / scraper_catalog.
    # scrape_company's closure marking). Cleared back to null if the
    # posting reappears in a later crawl -- see upsert_posting.
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    starred: Mapped[bool] = mapped_column(Boolean, default=False)
    starred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    note: Mapped[str | None] = mapped_column(Text)

    # The user's "not interested" signal -- distinct from closed_at (the
    # role fell off the board) and from starred (the opposite signal).
    # Excluded from Stage 2/3 batch submission and skill-trend evidence
    # unconditionally; hidden from GET /postings by default (see the
    # include_dismissed filter).
    dismissed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    company: Mapped["Company"] = relationship(back_populates="postings")
    verdicts: Mapped[list["Verdict"]] = relationship(
        back_populates="posting", order_by="Verdict.created_at.desc()"
    )
