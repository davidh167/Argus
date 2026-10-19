from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.utils.freshness import freshness_flag

ApplicationStatus = Literal["saved", "applied", "screening", "interview", "rejected", "skipped"]


class ApplicationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    posting_id: int
    posting_title: str
    posting_company_name: str
    # Reuses freshness_flag rather than a bare bool so "closed" isn't the
    # only state the desktop/extension could ever want to show here --
    # covers the 3.8 "warn on closed postings" requirement.
    posting_freshness: str
    status: str | None
    resume_used: str | None
    date_applied: date | None
    follow_up_date: date | None
    notes: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_application(cls, application) -> "ApplicationOut":
        # posting_title/posting_company_name/posting_freshness aren't
        # direct Application columns, so plain from_attributes can't
        # populate them -- single shared constructor so the router and the
        # MCP tool can't drift.
        posting = application.posting
        return cls(
            id=application.id,
            posting_id=application.posting_id,
            posting_title=posting.title,
            posting_company_name=posting.company.name,
            posting_freshness=freshness_flag(
                posting.posted_date, posting.first_seen_at, posting.closed_at
            ),
            status=application.status,
            resume_used=application.resume_used,
            date_applied=application.date_applied,
            follow_up_date=application.follow_up_date,
            notes=application.notes,
            created_at=application.created_at,
            updated_at=application.updated_at,
        )


class UpdateApplicationStatusRequest(BaseModel):
    status: ApplicationStatus
    resume_used: str | None = None
    date_applied: date | None = None
    follow_up_date: date | None = None
    notes: str | None = None
