from datetime import datetime

from sqlalchemy import Boolean, DateTime, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class SkillRoadmap(Base):
    __tablename__ = "skill_roadmap"

    id: Mapped[int] = mapped_column(primary_key=True)
    skill_or_language: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    # Free text, no DB enum -- matches Application.status / Company.ats_type's
    # convention. Null until a brief has been generated at least once.
    trigger_type: Mapped[str | None] = mapped_column(
        Text
    )  # 'job_market_gap'|'project_strengthener'
    is_new_territory: Mapped[bool | None] = mapped_column(Boolean)

    # Starred posting ids that surfaced this skill. JSONB list of ints, not
    # Postgres ARRAY -- every other structured column in this schema
    # (stage3_result, snapshot, extraction_strategy) is JSONB, so this stays
    # consistent rather than introducing the only ARRAY column for one field.
    # none_as_null=True: see Posting.locations for why -- applied to every
    # JSONB column in this schema after finding it broke that one live.
    trend_evidence: Mapped[list | None] = mapped_column(JSONB(none_as_null=True))

    # 'flagged'|'researching'|'in_progress'|'complete', no transition-order
    # validation -- mirrors update_application_status's permissiveness.
    status: Mapped[str] = mapped_column(Text, nullable=False, default="flagged")

    # Direction brief -- populated only once generate_roadmap_brief runs.
    what_it_is: Mapped[str | None] = mapped_column(Text)
    why_it_matters: Mapped[str | None] = mapped_column(Text)
    suggested_scope: Mapped[str | None] = mapped_column(Text)
    suggested_stack: Mapped[str | None] = mapped_column(Text)
    strengthens_project: Mapped[str | None] = mapped_column(Text)
    project_note: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
