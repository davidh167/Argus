from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from app.db import engine
from app.services import batches as batches_service
from app.services import key_skills_backfill
from app.services.postings import capture_posting
from app.services.profile import create_profile_version
from app.services.stage2 import Stage2UnavailableError

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


def make_fast_tracked_posting(db, company_name, url):
    # "ML Engineer" matches a Stage 1 fast-track pattern, so Stage 2 never runs
    # on it -- exactly the case the backfill exists for. Stage 2 is forced
    # unavailable anyway so a configured dev API key can't make a real call.
    with patch("app.services.postings.stage2_service.triage") as mock_triage:
        mock_triage.side_effect = Stage2UnavailableError("forced unavailable for test setup")
        posting, verdict = capture_posting(
            db, company_name=company_name, title="ML Engineer", raw_jd_text="build models", url=url
        )
    assert verdict.stage1_result == "fast_track"
    assert verdict.stage2_key_skills is None
    return posting, verdict


def test_list_pending_includes_fast_tracked_posting_without_key_skills(db):
    posting, _ = make_fast_tracked_posting(db, "KsCo1", "https://ksco1.example.com/1")
    assert posting.id in {p.id for p in key_skills_backfill.list_pending(db, [posting.id])}


def test_list_pending_excludes_postings_that_already_have_key_skills(db):
    posting, verdict = make_fast_tracked_posting(db, "KsCo2", "https://ksco2.example.com/1")
    verdict.stage2_key_skills = ["PyTorch"]
    db.flush()
    assert key_skills_backfill.list_pending(db, [posting.id]) == []


def test_list_pending_excludes_dismissed_postings(db):
    from datetime import datetime, timezone

    posting, _ = make_fast_tracked_posting(db, "KsCo3", "https://ksco3.example.com/1")
    posting.dismissed_at = datetime.now(timezone.utc)
    db.flush()
    assert key_skills_backfill.list_pending(db, [posting.id]) == []


def test_submit_records_a_backfill_kind_batch(db):
    posting, _ = make_fast_tracked_posting(db, "KsCo4", "https://ksco4.example.com/1")
    with patch.object(key_skills_backfill.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.key_skills_backfill.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.create.return_value = MagicMock(id="msgbatch_ks")
            mock_client_cls.return_value = mock_client
            result = key_skills_backfill.submit_backfill_batch(db, [posting.id])

    assert result["batch_id"] == "msgbatch_ks" and result["submitted"] == 1
    assert batches_service.get_batch_submission(db, "msgbatch_ks").kind == key_skills_backfill.KIND


def test_submit_with_nothing_pending_does_not_call_the_api(db):
    with patch.object(key_skills_backfill.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.key_skills_backfill.Anthropic") as mock_client_cls:
            result = key_skills_backfill.submit_backfill_batch(db, [])
    assert result["batch_id"] is None and result["submitted"] == 0
    mock_client_cls.assert_not_called()


def _fake_result(posting_id, key_skills, result="skip"):
    tool_use = MagicMock(
        type="tool_use", input={"result": result, "reason": "r", "key_skills": key_skills}
    )
    message = MagicMock(content=[tool_use], usage=MagicMock(input_tokens=100, output_tokens=20))
    return MagicMock(
        custom_id=f"posting-{posting_id}", result=MagicMock(type="succeeded", message=message)
    )


def test_apply_writes_only_key_skills_and_leaves_stage2_result_alone(db):
    posting, verdict = make_fast_tracked_posting(db, "KsCo5", "https://ksco5.example.com/1")
    with patch.object(key_skills_backfill.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.key_skills_backfill.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.retrieve.return_value = MagicMock(
                processing_status="ended"
            )
            # Stage 2 would have said "skip" -- it must NOT be written onto a fast-tracked posting.
            mock_client.messages.batches.results.return_value = [
                _fake_result(posting.id, ["PyTorch", "RAG"])
            ]
            mock_client_cls.return_value = mock_client
            result = key_skills_backfill.apply_backfill_results(db, "msgbatch_ks")

    assert result["applied"] == 1 and result["posting_ids"] == [posting.id]
    db.refresh(verdict)
    assert verdict.stage2_key_skills == ["PyTorch", "RAG"]
    assert verdict.stage2_result is None
    assert verdict.stage2_tokens_used is None


def test_apply_does_not_overwrite_key_skills_written_in_the_meantime(db):
    posting, verdict = make_fast_tracked_posting(db, "KsCo6", "https://ksco6.example.com/1")
    verdict.stage2_key_skills = ["Already", "There"]
    db.flush()
    with patch.object(key_skills_backfill.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.key_skills_backfill.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.retrieve.return_value = MagicMock(
                processing_status="ended"
            )
            mock_client.messages.batches.results.return_value = [_fake_result(posting.id, ["New"])]
            mock_client_cls.return_value = mock_client
            result = key_skills_backfill.apply_backfill_results(db, "msgbatch_ks")

    assert result["applied"] == 0 and result["skipped_stale"] == 1
    db.refresh(verdict)
    assert verdict.stage2_key_skills == ["Already", "There"]


def test_apply_returns_status_without_applying_while_batch_in_progress(db):
    with patch.object(key_skills_backfill.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.key_skills_backfill.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.retrieve.return_value = MagicMock(
                processing_status="in_progress"
            )
            mock_client_cls.return_value = mock_client
            result = key_skills_backfill.apply_backfill_results(db, "msgbatch_ks")
    assert result["status"] == "in_progress" and result["applied"] == 0
    mock_client.messages.batches.results.assert_not_called()


# -- refresh (re-extract with the current prompt, interest set only) --


def test_list_pending_refresh_includes_postings_that_already_have_key_skills(db):
    posting, verdict = make_fast_tracked_posting(db, "KsRef1", "https://ksref1.example.com/1")
    verdict.stage2_key_skills = ["old style skill"]
    db.flush()
    assert posting.id not in {p.id for p in key_skills_backfill.list_pending(db, [posting.id])}
    assert posting.id in {
        p.id for p in key_skills_backfill.list_pending(db, [posting.id], refresh=True)
    }


def _submit_refresh(db, posting_ids):
    with patch.object(key_skills_backfill.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.key_skills_backfill.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.create.return_value = MagicMock(id="msgbatch_ref")
            mock_client_cls.return_value = mock_client
            return key_skills_backfill.submit_refresh_batch(db, posting_ids)


def test_refresh_submits_only_interest_set_postings_with_its_own_kind(db):
    picked, v1 = make_fast_tracked_posting(
        db, "KsRef2", "https://ksref2.example.com/1"
    )  # manual capture
    bulk, v2 = make_fast_tracked_posting(db, "KsRef3", "https://ksref3.example.com/1")
    bulk.source = "scraped"
    bulk.detail_fetched_at = datetime.now(timezone.utc)
    v1.stage2_key_skills, v2.stage2_key_skills = ["old"], ["old"]
    db.flush()
    result = _submit_refresh(db, [picked.id, bulk.id])
    assert result["submitted"] == 1
    assert (
        batches_service.get_batch_submission(db, "msgbatch_ref").kind
        == key_skills_backfill.KIND_REFRESH
    )


def test_apply_refresh_overwrites_existing_key_skills(db):
    posting, verdict = make_fast_tracked_posting(db, "KsRef4", "https://ksref4.example.com/1")
    verdict.stage2_key_skills = ["Old Style Skill"]
    db.flush()
    with patch.object(key_skills_backfill.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.key_skills_backfill.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.retrieve.return_value = MagicMock(
                processing_status="ended"
            )
            mock_client.messages.batches.results.return_value = [
                _fake_result(posting.id, ["PyTorch"])
            ]
            mock_client_cls.return_value = mock_client
            result = key_skills_backfill.apply_refresh_results(db, "msgbatch_ref")
    assert result["applied"] == 1 and result["skipped_stale"] == 0
    db.refresh(verdict)
    assert verdict.stage2_key_skills == ["PyTorch"]
