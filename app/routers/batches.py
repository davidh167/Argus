from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas.batch import BatchSubmissionOut
from app.services import batches as batches_service

router = APIRouter(prefix="/batches", tags=["batches"])


@router.get("", response_model=list[BatchSubmissionOut])
def list_batches(kind: str | None = None, db: Session = Depends(get_db)):
    return batches_service.list_batch_submissions(db, kind=kind)
