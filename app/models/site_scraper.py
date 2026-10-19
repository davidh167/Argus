from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class SiteScraper(Base):
    __tablename__ = "site_scrapers"

    id: Mapped[int] = mapped_column(primary_key=True)
    domain: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    ats_type: Mapped[str | None] = mapped_column(Text)
    # none_as_null=True: see Posting.locations for why -- applied to every
    # JSONB column in this schema after finding it broke that one live.
    extraction_strategy: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True))
    generated_by: Mapped[str | None] = mapped_column(Text)  # 'llm'|'manual'
    last_validated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    success_rate: Mapped[float | None] = mapped_column(Float)
    fallback_count: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[str | None] = mapped_column(Text)
