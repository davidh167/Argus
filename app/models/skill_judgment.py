from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class SkillJudgment(Base):
    """An LLM's have/developing/gap call for one market-demand skill against
    one profile version (app/services/skill_judgment.py). Cached because the
    answer only changes when the profile does; market_demand prefers it over
    its embedding-similarity guess, which can't tell an umbrella skill the
    profile demonstrates ("LLMs") from an unrelated gap ("C/C++")."""

    __tablename__ = "skill_judgments"
    __table_args__ = (UniqueConstraint("profile_version_id", "skill_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    profile_version_id: Mapped[int] = mapped_column(
        ForeignKey("profile_versions.id"), nullable=False
    )
    # roadmap._normalize_phrase(skill_label): case/hyphen-insensitive identity.
    skill_key: Mapped[str] = mapped_column(Text, nullable=False)
    skill_label: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)  # have|developing|gap
    # Text of the profile evidence items the model cited (empty for a gap).
    evidence: Mapped[list] = mapped_column(JSONB(none_as_null=True), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
