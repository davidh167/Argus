import csv
import io

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas.application import ApplicationOut, UpdateApplicationStatusRequest
from app.services import applications as applications_service
from app.services import postings as postings_service

# No prefix: this resource's two endpoints span different URL namespaces
# (nested under /postings for the upsert action, flat /applications for the
# list) -- a deliberate one-off rather than two separate routers/
# include_router calls, which would double the surface for forgetting the
# auth dependency in app/main.py.
router = APIRouter(tags=["applications"])

_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(cell):
    """Neutralize spreadsheet formula injection: posting titles/company names are attacker-controlled."""
    if isinstance(cell, str) and cell.startswith(_CSV_FORMULA_PREFIXES):
        return "'" + cell
    return cell


@router.post("/postings/{posting_id}/application", response_model=ApplicationOut)
def update_application_status(
    posting_id: int, payload: UpdateApplicationStatusRequest, db: Session = Depends(get_db)
):
    if postings_service.get_posting(db, posting_id) is None:
        raise HTTPException(status_code=404, detail="Posting not found")
    application = applications_service.update_application_status(
        db,
        posting_id,
        payload.status,
        resume_used=payload.resume_used,
        date_applied=payload.date_applied,
        follow_up_date=payload.follow_up_date,
        notes=payload.notes,
    )
    db.commit()
    db.refresh(application)
    return ApplicationOut.from_application(application)


@router.get("/applications", response_model=list[ApplicationOut])
def list_applications(
    status: str | None = None, sort: str = "updated", db: Session = Depends(get_db)
):
    """sort is 'updated' (default, most recently touched first) or
    'follow_up' (soonest follow_up_date first, nulls last)."""
    return [
        ApplicationOut.from_application(a)
        for a in applications_service.list_applications(db, status=status, sort=sort)
    ]


@router.get("/applications/export.csv")
def export_applications_csv(
    status: str | None = None, sort: str = "updated", db: Session = Depends(get_db)
):
    rows = [
        ApplicationOut.from_application(a)
        for a in applications_service.list_applications(db, status=status, sort=sort)
    ]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "posting_title",
            "posting_company_name",
            "posting_freshness",
            "status",
            "resume_used",
            "date_applied",
            "follow_up_date",
            "notes",
        ]
    )
    for r in rows:
        writer.writerow(
            _csv_safe(cell)
            for cell in [
                r.posting_title,
                r.posting_company_name,
                r.posting_freshness,
                r.status or "",
                r.resume_used or "",
                r.date_applied or "",
                r.follow_up_date or "",
                r.notes or "",
            ]
        )
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=applications.csv"},
    )


@router.get("/postings/{posting_id}/application", response_model=ApplicationOut | None)
def get_application(posting_id: int, db: Session = Depends(get_db)):
    if postings_service.get_posting(db, posting_id) is None:
        raise HTTPException(status_code=404, detail="Posting not found")
    application = applications_service.get_application_for_posting(db, posting_id)
    return ApplicationOut.from_application(application) if application else None
