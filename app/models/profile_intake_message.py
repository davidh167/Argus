from datetime import datetime

from sqlalchemy import DateTime, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class ProfileIntakeMessage(Base):
    """One turn of the profile-intake chat (app/services/profile_intake.py)
    -- a single ongoing conversation, not multiple named sessions (this is
    a single-user local app with one profile to reconcile against, so
    there's nothing to disambiguate between). Append-only log, ordered by
    id; GET /profile/intake-chat/history returns it in full so the desktop
    chat panel can resume exactly where a prior session left off, and
    DELETE /profile/intake-chat/history clears it to start a fresh
    conversation on request."""

    __tablename__ = "profile_intake_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    # 'user'|'assistant', no DB enum -- matches Application.status/
    # Company.ats_type's established free-text convention.
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
