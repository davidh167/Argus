from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class BatchSubmission(Base):
    __tablename__ = "batch_submissions"

    id: Mapped[int] = mapped_column(primary_key=True)
    batch_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    # Free text, no DB enum -- matches Posting.source / Application.status's
    # convention. 'stage2_triage'|'stage3_analysis'.
    kind: Mapped[str] = mapped_column(Text, nullable=False)

    submitted_count: Mapped[int] = mapped_column(Integer, nullable=False)
    capped: Mapped[bool] = mapped_column(Boolean, nullable=False)
    company_id: Mapped[int | None] = mapped_column(ForeignKey("companies.id"))

    # The profile version every request in this batch was built against.
    # Only meaningful (and only ever set) for kind='stage3_analysis': that
    # submit path fetches ONE current profile snapshot and reuses it for
    # every request in the batch. Stage 2 has no single batch-wide answer --
    # each request's criteria come from that posting's own verdict, which
    # can differ posting to posting -- so it's left null for
    # kind='stage2_triage'. Consumed by stage3_batch.apply_batch_results to
    # stamp Verdict.stage3_profile_version_id once results come back,
    # possibly long after submission.
    profile_version_id: Mapped[int | None] = mapped_column(ForeignKey("profile_versions.id"))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Ids of the postings this batch was built for, so eligibility queries can
    # exclude postings already "in flight" (otherwise a submitted-but-unapplied
    # batch's postings still look pending and get paid for twice). Null for
    # batches recorded before this column existed.
    posting_ids: Mapped[list | None] = mapped_column(JSONB(none_as_null=True))

    # Set once the batch's results have been applied (it had ended). A batch
    # with applied_at null and posting_ids set blocks resubmission of its postings.
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Cached from the last time anyone checked -- not live, refreshed only
    # when apply_batch_results actually calls the Anthropic API.
    last_status: Mapped[str | None] = mapped_column(Text)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
