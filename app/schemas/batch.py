from datetime import datetime

from pydantic import BaseModel, ConfigDict


class BatchSubmissionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    batch_id: str
    kind: str
    submitted_count: int
    capped: bool
    company_id: int | None
    created_at: datetime
    last_status: str | None
    last_checked_at: datetime | None
    applied_at: datetime | None = None
