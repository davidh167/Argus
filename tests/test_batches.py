import pytest
from sqlalchemy.orm import Session

from app.db import engine
from app.services import batches
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


def test_record_batch_submission_returns_a_row_with_the_given_fields(db):
    submission = batches.record_batch_submission(
        db,
        "msgbatch_test1",
        kind="stage3_analysis",
        submitted_count=5,
        capped=True,
        company_id=None,
    )
    assert submission.batch_id == "msgbatch_test1"
    assert submission.kind == "stage3_analysis"
    assert submission.submitted_count == 5
    assert submission.capped is True
    assert submission.last_status is None


def test_list_batch_submissions_filters_by_kind(db):
    batches.record_batch_submission(
        db, "msgbatch_test2", kind="stage2_triage", submitted_count=3, capped=False
    )
    batches.record_batch_submission(
        db, "msgbatch_test3", kind="stage3_analysis", submitted_count=2, capped=False
    )

    stage2_only = batches.list_batch_submissions(db, kind="stage2_triage")
    stage2_ids = {s.batch_id for s in stage2_only}
    assert "msgbatch_test2" in stage2_ids
    assert "msgbatch_test3" not in stage2_ids


def test_list_batch_submissions_orders_newest_first(db):
    first = batches.record_batch_submission(
        db, "msgbatch_test4", kind="stage2_triage", submitted_count=1, capped=False
    )
    second = batches.record_batch_submission(
        db, "msgbatch_test5", kind="stage2_triage", submitted_count=1, capped=False
    )

    ids_in_order = [
        s.batch_id
        for s in batches.list_batch_submissions(db, kind="stage2_triage")
        if s.batch_id in {first.batch_id, second.batch_id}
    ]
    assert ids_in_order == [second.batch_id, first.batch_id]


def test_update_batch_status_updates_existing_row(db):
    submission = batches.record_batch_submission(
        db, "msgbatch_test6", kind="stage3_analysis", submitted_count=1, capped=False
    )
    assert submission.last_status is None

    batches.update_batch_status(db, "msgbatch_test6", "ended")
    db.refresh(submission)
    assert submission.last_status == "ended"
    assert submission.last_checked_at is not None


def test_update_batch_status_noops_for_unknown_batch_id(db):
    batches.update_batch_status(db, "msgbatch_does_not_exist", "ended")  # must not raise


def test_record_batch_submission_accepts_profile_version_id(db):
    from app.services.profile import get_current_profile

    current = get_current_profile(db)
    submission = batches.record_batch_submission(
        db,
        "msgbatch_test7",
        kind="stage3_analysis",
        submitted_count=1,
        capped=False,
        profile_version_id=current.id,
    )
    assert submission.profile_version_id == current.id


def test_record_batch_submission_defaults_profile_version_id_to_none(db):
    # stage2_triage batches have no single meaningful profile version (each
    # request's criteria can come from a different posting's own verdict).
    submission = batches.record_batch_submission(
        db, "msgbatch_test8", kind="stage2_triage", submitted_count=1, capped=False
    )
    assert submission.profile_version_id is None


def test_get_batch_submission_returns_the_matching_row(db):
    batches.record_batch_submission(
        db, "msgbatch_test9", kind="stage2_triage", submitted_count=1, capped=False
    )
    found = batches.get_batch_submission(db, "msgbatch_test9")
    assert found is not None
    assert found.batch_id == "msgbatch_test9"


def test_get_batch_submission_returns_none_for_unknown_batch_id(db):
    assert batches.get_batch_submission(db, "msgbatch_does_not_exist") is None
