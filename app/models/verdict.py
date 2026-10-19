from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Verdict(Base):
    __tablename__ = "verdicts"

    id: Mapped[int] = mapped_column(primary_key=True)
    posting_id: Mapped[int] = mapped_column(ForeignKey("postings.id"), index=True)
    profile_version_id: Mapped[int] = mapped_column(ForeignKey("profile_versions.id"))

    # Superseded by the computed stage2_stale/stage3_stale (see
    # app/schemas/verdict.py's build_verdict_out). This column was never
    # written anywhere, so every row reads False regardless of real
    # staleness; nothing reads it anymore and it is kept only to avoid a
    # destructive migration.
    is_stale: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Stage 1 (rules-based, title only)
    stage1_result: Mapped[str | None] = mapped_column(Text)  # skip|fast_track|analyze
    stage1_reason: Mapped[str | None] = mapped_column(Text)

    # Stage 2 (Haiku triage) -- columns exist now, pipeline logic deferred
    stage2_result: Mapped[str | None] = mapped_column(Text)
    stage2_reason: Mapped[str | None] = mapped_column(Text)
    stage2_tokens_used: Mapped[int | None] = mapped_column(Integer)
    # At most stage2.MAX_KEY_SKILLS short noun phrases naming specific
    # technologies/tools this posting requires (4.3) -- feeds
    # roadmap.market_demand, independent of this verdict's fit result.
    # none_as_null=True: see Posting.locations for why -- applied to every
    # JSONB column in this schema.
    stage2_key_skills: Mapped[list | None] = mapped_column(JSONB(none_as_null=True))
    # Profile version the stage2 criteria were actually built against
    # (sync and batch both source criteria from THIS verdict's own
    # profile_version_id, so it's always set to that same value -- see
    # run_stage2_triage and stage2_batch.apply_batch_results). Null on
    # rows written before this column existed.
    stage2_profile_version_id: Mapped[int | None] = mapped_column(ForeignKey("profile_versions.id"))

    # Stage 3 (Sonnet structured fit, manually triggered) -- deferred
    stage3_triggered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # none_as_null=True: see Posting.locations for why -- applied to every
    # JSONB column in this schema after finding it broke that one live.
    stage3_result: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True))
    stage3_tokens_used: Mapped[int | None] = mapped_column(Integer)
    # Profile version stage3 was actually run against -- the CURRENT profile
    # at trigger time, not this verdict's own profile_version_id (Stage 3
    # has no stage1/stage2 gate and always uses get_current_profile). For
    # the sync path that's known directly; for the batch path it comes from
    # the tracked BatchSubmission.profile_version_id recorded at submit
    # time (see stage3_batch.py). Null when provenance isn't known (legacy
    # rows, or an untracked batch_id) -- treated as "not stale" rather than
    # guessed, per this codebase's no-speculative-fallbacks convention.
    stage3_profile_version_id: Mapped[int | None] = mapped_column(ForeignKey("profile_versions.id"))

    posting: Mapped["Posting"] = relationship(back_populates="verdicts")
