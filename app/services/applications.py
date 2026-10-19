from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.application import Application


def get_or_create_application(db: Session, posting_id: int) -> Application:
    application = db.scalar(select(Application).where(Application.posting_id == posting_id))
    if application is not None:
        return application
    application = Application(posting_id=posting_id)
    db.add(application)
    db.flush()
    return application


def get_application_for_posting(db: Session, posting_id: int) -> Application | None:
    """Plain read, no side effect -- unlike get_or_create_application, this
    must not insert a bare row just because someone checked."""
    return db.scalar(select(Application).where(Application.posting_id == posting_id))


def update_application_status(
    db: Session,
    posting_id: int,
    status: str,
    *,
    resume_used: str | None = None,
    date_applied: date | None = None,
    follow_up_date: date | None = None,
    notes: str | None = None,
) -> Application:
    """Upsert semantics: get-or-create the posting's one application row,
    then set status unconditionally (that's the point of calling this), and
    set each optional field ONLY when the caller passed a non-None value
    this call -- omitting a kwarg never clobbers a value stored by a
    previous call. There is no way to explicitly clear a field back to NULL
    through this function; nothing asks for that."""
    application = get_or_create_application(db, posting_id)
    application.status = status
    if resume_used is not None:
        application.resume_used = resume_used
    if date_applied is not None:
        application.date_applied = date_applied
    elif status == "applied" and application.date_applied is None:
        # Auto-set only on the transition to 'applied' and only when
        # nothing's there yet -- never clobbers a manually-entered or
        # previously auto-set date on a later status change (e.g.
        # applied -> screening -> applied again shouldn't reset it).
        application.date_applied = date.today()
    if follow_up_date is not None:
        application.follow_up_date = follow_up_date
    if notes is not None:
        application.notes = notes
    db.flush()
    return application


def list_applications(
    db: Session, status: str | None = None, sort: str = "updated"
) -> list[Application]:
    stmt = select(Application)
    if status is not None:
        stmt = stmt.where(Application.status == status)
    if sort == "follow_up":
        stmt = stmt.order_by(Application.follow_up_date.asc().nulls_last())
    else:
        stmt = stmt.order_by(Application.updated_at.desc())
    return list(db.scalars(stmt))
