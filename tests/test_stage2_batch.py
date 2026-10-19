from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from app.db import engine
from app.services import stage2_batch
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


def make_analyze_posting(db, company_name, url, raw_jd_text="generic role"):
    # "Software Engineer II" falls through Stage 1 to 'analyze', per test_stage1.py.
    # capture_posting auto-runs Stage 2 when a real API key is configured (which it
    # is in this dev environment) -- these tests need postings still PENDING Stage 2
    # to exercise the batch path, so Stage 2 is forced unavailable during creation.
    with patch("app.services.postings.stage2_service.triage") as mock_triage:
        mock_triage.side_effect = stage2_batch.Stage2UnavailableError(
            "forced unavailable for test setup"
        )
        posting, verdict = capture_posting(
            db,
            company_name=company_name,
            title="Software Engineer II",
            raw_jd_text=raw_jd_text,
            url=url,
        )
    assert verdict.stage1_result == "analyze"
    assert verdict.stage2_result is None
    return posting, verdict


def test_submit_pending_batch_builds_one_request_per_posting_with_custom_id(db):
    p1, _ = make_analyze_posting(db, "BatchCo1", "https://batchco1.example.com/1")
    p2, _ = make_analyze_posting(db, "BatchCo2", "https://batchco2.example.com/1")

    with patch.object(stage2_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage2_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_batch = MagicMock(id="msgbatch_123")
            mock_client.messages.batches.create.return_value = mock_batch
            mock_client_cls.return_value = mock_client

            result = stage2_batch.submit_pending_batch(db, posting_ids=[p1.id, p2.id])

    assert result == {
        "batch_id": "msgbatch_123",
        "submitted": 2,
        "capped": False,
        "excluded_unhydrated": 0,
    }
    submitted_requests = mock_client.messages.batches.create.call_args.kwargs["requests"]
    custom_ids = {r["custom_id"] for r in submitted_requests}
    assert custom_ids == {f"posting-{p1.id}", f"posting-{p2.id}"}

    from app.services import batches as batches_service

    tracked = [
        s for s in batches_service.list_batch_submissions(db) if s.batch_id == "msgbatch_123"
    ]
    assert len(tracked) == 1
    assert tracked[0].kind == "stage2_triage"
    assert tracked[0].submitted_count == 2


def test_submit_pending_batch_passes_education_and_skills_into_each_request(db):
    # Regression test, batch-path equivalent of
    # test_postings_service.test_capture_posting_passes_education_and_skills_to_stage2_triage.
    snapshot = {
        **EMPTY_SNAPSHOT,
        "credentials": {
            "education": ["MS Computer Science, Cal Poly SLO"],
            "oss_contributions": [],
            "publications": [],
        },
        "skills": {"core": ["PyTorch"], "developing": ["Kubernetes"], "coursework": []},
    }
    create_profile_version(db, snapshot=snapshot)
    p1, _ = make_analyze_posting(db, "EducationBatchCo", "https://educationbatchco.example.com/1")

    with patch.object(stage2_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage2_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.create.return_value = MagicMock(id="msgbatch_education")
            mock_client_cls.return_value = mock_client
            stage2_batch.submit_pending_batch(db, posting_ids=[p1.id])

    submitted_requests = mock_client.messages.batches.create.call_args.kwargs["requests"]
    sent_content = submitted_requests[0]["params"]["messages"][0]["content"]
    assert "MS Computer Science, Cal Poly SLO" in sent_content
    assert "PyTorch" in sent_content
    assert "Kubernetes" in sent_content


def make_unhydrated_scraped_posting(db, company_name, url):
    """A scraped posting whose title falls to 'analyze' but never got a
    detail-page fetch -- source='scraped', detail_fetched_at is null."""
    from app.services.companies import get_or_create_company
    from app.services.postings import upsert_posting, write_stage1_verdict

    company = get_or_create_company(db, company_name)
    posting, *_ = upsert_posting(
        db,
        company,
        title="Software Engineer II",
        url=url,
        external_id=None,
        raw_jd_text="short blurb",
        posted_date=None,
        source="scraped",
    )
    verdict = write_stage1_verdict(db, posting)
    assert verdict.stage1_result == "analyze"
    return posting


def test_submit_pending_batch_excludes_unhydrated_scraped_postings(db):
    hydrated_eligible, _ = make_analyze_posting(
        db, "HydratedCo", "https://hydratedco.example.com/1"
    )
    unhydrated = make_unhydrated_scraped_posting(
        db, "UnhydratedCo", "https://unhydratedco.example.com/1"
    )

    with patch.object(stage2_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage2_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.create.return_value = MagicMock(id="msgbatch_excl")
            mock_client_cls.return_value = mock_client

            result = stage2_batch.submit_pending_batch(
                db, posting_ids=[hydrated_eligible.id, unhydrated.id]
            )

    assert result["submitted"] == 1
    assert result["excluded_unhydrated"] == 1
    submitted_requests = mock_client.messages.batches.create.call_args.kwargs["requests"]
    submitted_ids = {r["custom_id"] for r in submitted_requests}
    assert submitted_ids == {f"posting-{hydrated_eligible.id}"}


def test_submit_pending_batch_returns_none_batch_id_when_nothing_pending(db):
    with patch.object(stage2_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage2_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            result = stage2_batch.submit_pending_batch(db, posting_ids=[999999])

    assert result == {"batch_id": None, "submitted": 0, "capped": False, "excluded_unhydrated": 0}
    mock_client.messages.batches.create.assert_not_called()


def test_submit_pending_batch_raises_stage2_unavailable_without_api_key(db):
    with patch.object(stage2_batch.settings, "anthropic_api_key", None):
        with patch("app.services.stage2_batch.Anthropic") as mock_client_cls:
            with pytest.raises(stage2_batch.Stage2UnavailableError):
                stage2_batch.submit_pending_batch(db)
            mock_client_cls.assert_not_called()


def _fake_succeeded_item(
    custom_id: str, result: str, reason: str, key_skills: list[str] | None = None
):
    tool_use_block = MagicMock(
        type="tool_use", input={"result": result, "reason": reason, "key_skills": key_skills or []}
    )
    message = MagicMock(content=[tool_use_block], usage=MagicMock(input_tokens=10, output_tokens=5))
    return MagicMock(custom_id=custom_id, result=MagicMock(type="succeeded", message=message))


def test_apply_batch_results_maps_results_back_to_correct_posting(db):
    p1, _ = make_analyze_posting(db, "BatchCo3", "https://batchco3.example.com/1")
    p2, _ = make_analyze_posting(db, "BatchCo4", "https://batchco4.example.com/1")

    items = [
        _fake_succeeded_item(
            f"posting-{p1.id}", "fast_track", "reason for p1", key_skills=["Rust"]
        ),
        _fake_succeeded_item(f"posting-{p2.id}", "skip", "reason for p2"),
    ]
    with patch.object(stage2_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage2_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.retrieve.return_value = MagicMock(
                processing_status="ended"
            )
            mock_client.messages.batches.results.return_value = items
            mock_client_cls.return_value = mock_client

            result = stage2_batch.apply_batch_results(db, "msgbatch_123")

    assert result["status"] == "ended"
    assert result["applied"] == 2
    assert result["errors"] == 0
    assert result["skipped_stale"] == 0
    assert set(result["posting_ids"]) == {p1.id, p2.id}
    db.refresh(p1.verdicts[0])
    db.refresh(p2.verdicts[0])
    assert p1.verdicts[0].stage2_reason == "reason for p1"
    assert p2.verdicts[0].stage2_reason == "reason for p2"
    assert p1.verdicts[0].stage2_key_skills == ["Rust"]
    # Correct provenance: the verdict's own profile_version_id, since
    # that's what the criteria were built from at submit time (both for
    # this sync-looking batch path and the original request).
    assert p1.verdicts[0].stage2_profile_version_id == p1.verdicts[0].profile_version_id
    assert p2.verdicts[0].stage2_profile_version_id == p2.verdicts[0].profile_version_id


def test_apply_batch_results_updates_tracked_submission_status(db):
    from app.services import batches as batches_service

    batches_service.record_batch_submission(
        db, "msgbatch_tracked", kind="stage2_triage", submitted_count=0, capped=False
    )
    with patch.object(stage2_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage2_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.retrieve.return_value = MagicMock(
                processing_status="in_progress"
            )
            mock_client_cls.return_value = mock_client

            stage2_batch.apply_batch_results(db, "msgbatch_tracked")

    tracked = [
        s for s in batches_service.list_batch_submissions(db) if s.batch_id == "msgbatch_tracked"
    ]
    assert tracked[0].last_status == "in_progress"
    assert tracked[0].last_checked_at is not None


def test_apply_batch_results_reports_in_progress_status_without_applying(db):
    with patch.object(stage2_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage2_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.retrieve.return_value = MagicMock(
                processing_status="in_progress"
            )
            mock_client_cls.return_value = mock_client

            result = stage2_batch.apply_batch_results(db, "msgbatch_123")

    assert result == {
        "status": "in_progress",
        "applied": 0,
        "errors": 0,
        "skipped_stale": 0,
        "posting_ids": [],
    }
    mock_client.messages.batches.results.assert_not_called()


def test_apply_batch_results_counts_errored_without_raising(db):
    p1, _ = make_analyze_posting(db, "BatchCo5", "https://batchco5.example.com/1")
    p2, _ = make_analyze_posting(db, "BatchCo6", "https://batchco6.example.com/1")

    errored_item = MagicMock(
        custom_id=f"posting-{p2.id}",
        result=MagicMock(type="errored", error=MagicMock()),
    )
    items = [_fake_succeeded_item(f"posting-{p1.id}", "analyze", "ok"), errored_item]
    with patch.object(stage2_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage2_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.retrieve.return_value = MagicMock(
                processing_status="ended"
            )
            mock_client.messages.batches.results.return_value = items
            mock_client_cls.return_value = mock_client

            result = stage2_batch.apply_batch_results(db, "msgbatch_123")

    assert result["status"] == "ended"
    assert result["applied"] == 1
    assert result["errors"] == 1
    assert result["skipped_stale"] == 0
    assert result["posting_ids"] == [p1.id]


def test_apply_batch_results_still_applies_when_recrawl_created_a_same_shape_verdict(db):
    p1, verdict = make_analyze_posting(db, "BatchCo7", "https://batchco7.example.com/1")
    # A recrawl during the batch's flight time always appends a new verdict
    # row (write_stage1_verdict never updates in place). If that new latest
    # verdict still has the same eligible shape (analyze, no stage2 yet),
    # applying the batch's result to it is correct, not stale.
    from app.services.postings import write_stage1_verdict

    write_stage1_verdict(db, p1)

    items = [_fake_succeeded_item(f"posting-{p1.id}", "fast_track", "still applies")]
    with patch.object(stage2_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage2_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.retrieve.return_value = MagicMock(
                processing_status="ended"
            )
            mock_client.messages.batches.results.return_value = items
            mock_client_cls.return_value = mock_client

            result = stage2_batch.apply_batch_results(db, "msgbatch_123")

    assert result["applied"] == 1


def test_apply_batch_results_skips_stale_verdict_shape_changed(db):
    p1, verdict = make_analyze_posting(db, "BatchCo8", "https://batchco8.example.com/1")
    # Simulate the posting having already been triaged by a different path
    # (e.g. manual capture re-run) between submission and result application
    verdict.stage2_result = "analyze"
    verdict.stage2_reason = "already triaged elsewhere"
    db.flush()

    items = [_fake_succeeded_item(f"posting-{p1.id}", "fast_track", "stale reason")]
    with patch.object(stage2_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage2_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.retrieve.return_value = MagicMock(
                processing_status="ended"
            )
            mock_client.messages.batches.results.return_value = items
            mock_client_cls.return_value = mock_client

            result = stage2_batch.apply_batch_results(db, "msgbatch_123")

    assert result == {
        "status": "ended",
        "applied": 0,
        "errors": 0,
        "skipped_stale": 1,
        "posting_ids": [],
    }
    assert verdict.stage2_reason == "already triaged elsewhere"  # untouched


def test_submit_pending_batch_caps_and_ranks_by_relevance(db):
    create_profile_version(
        db,
        snapshot={
            **EMPTY_SNAPSHOT,
            "skills": {
                "core": ["kubernetes", "distributed", "systems"],
                "developing": [],
                "coursework": [],
            },
            "criteria": {
                **EMPTY_SNAPSHOT["criteria"],
                "target_signals": ["kubernetes", "distributed systems"],
            },
        },
    )
    p_relevant, _ = make_analyze_posting(
        db,
        "RelCo1",
        "https://relco1.example.com/1",
        raw_jd_text="Build distributed systems on kubernetes at scale.",
    )
    p_irrelevant, _ = make_analyze_posting(
        db,
        "RelCo2",
        "https://relco2.example.com/1",
        raw_jd_text="Sales representative, enterprise accounts, quota driven.",
    )

    with patch.object(stage2_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage2_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.create.return_value = MagicMock(id="msgbatch_123")
            mock_client_cls.return_value = mock_client

            result = stage2_batch.submit_pending_batch(
                db, posting_ids=[p_relevant.id, p_irrelevant.id], limit=1
            )

    assert result == {
        "batch_id": "msgbatch_123",
        "submitted": 1,
        "capped": True,
        "excluded_unhydrated": 0,
    }
    submitted_requests = mock_client.messages.batches.create.call_args.kwargs["requests"]
    submitted_ids = {r["custom_id"] for r in submitted_requests}
    assert submitted_ids == {f"posting-{p_relevant.id}"}


def test_submit_pending_batch_uncapped_when_limit_is_none(db):
    p1, _ = make_analyze_posting(db, "RelCo3", "https://relco3.example.com/1")
    p2, _ = make_analyze_posting(db, "RelCo4", "https://relco4.example.com/1")

    with patch.object(stage2_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage2_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.create.return_value = MagicMock(id="msgbatch_123")
            mock_client_cls.return_value = mock_client

            result = stage2_batch.submit_pending_batch(db, posting_ids=[p1.id, p2.id], limit=None)

    assert result == {
        "batch_id": "msgbatch_123",
        "submitted": 2,
        "capped": False,
        "excluded_unhydrated": 0,
    }


def test_stage2_excludes_postings_in_an_unapplied_batch(db):
    from app.services import batches as batches_service
    from app.services.postings import list_stage2_pending

    posting, _ = make_analyze_posting(db, "S2InFlightCo", "https://s2inflightco.example.com/1")
    assert [p.id for p in list_stage2_pending(db, posting_ids=[posting.id])] == [posting.id]

    batches_service.record_batch_submission(
        db,
        "msgbatch_s2_inflight",
        kind="stage2_triage",
        submitted_count=1,
        capped=False,
        posting_ids=[posting.id],
    )
    assert list_stage2_pending(db, posting_ids=[posting.id]) == []

    batches_service.mark_applied(db, "msgbatch_s2_inflight")
    assert [p.id for p in list_stage2_pending(db, posting_ids=[posting.id])] == [posting.id]
