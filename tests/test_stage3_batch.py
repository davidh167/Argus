from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from app.db import engine
from app.services import stage3_batch
from app.services.postings import capture_posting
from app.services.profile import create_profile_version
from tests.conftest import TITLE_RULES

EMPTY_SNAPSHOT = {
    "experience": [],
    "projects": [],
    "skills": {"core": [], "developing": [], "coursework": []},
    "credentials": {"education": [], "oss_contributions": [], "publications": []},
    "criteria": {
        "skip_patterns": [],
        "target_signals": [],
        **TITLE_RULES,
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


def make_pending_posting(db, company_name, url, raw_jd_text="generic role"):
    # "ML Engineer" fast-tracks at Stage 1, so capture_posting never attempts
    # a real Stage 2 call -- Stage 3 never runs automatically on capture
    # regardless, so no need to force-fail anything here (unlike Stage 2's
    # equivalent fixture).
    posting, verdict = capture_posting(
        db, company_name=company_name, title="ML Engineer", raw_jd_text=raw_jd_text, url=url
    )
    assert verdict.stage3_result is None
    return posting, verdict


def make_unhydrated_scraped_posting(db, company_name, url):
    """A scraped posting that never got a detail-page fetch --
    source='scraped', detail_fetched_at is null. Stage 3 has no stage1/
    stage2 gate, so any title works here."""
    from app.services.companies import get_or_create_company
    from app.services.postings import upsert_posting, write_stage1_verdict

    company = get_or_create_company(db, company_name)
    posting, *_ = upsert_posting(
        db,
        company,
        title="ML Engineer",
        url=url,
        external_id=None,
        raw_jd_text="short blurb",
        posted_date=None,
        source="scraped",
    )
    write_stage1_verdict(db, posting)
    return posting


FAKE_STAGE3_INPUT = {
    "verdict": "Strong Fit",
    "seniority_match": "Yes",
    "domain_match": "Yes",
    "stack_overlap": ["Python"],
    "hard_gaps": [],
    "soft_gaps": [],
    "developing_gaps": [],
    "roadmap_gaps": [],
    "roadmap_gap_caveat": None,
    "fde_flag": False,
    "resume_base": "ml_search.tex",
    "one_liner": "Strong match.",
}


def test_submit_pending_batch_builds_one_request_per_posting_with_custom_id(db):
    p1, _ = make_pending_posting(db, "S3BatchCo1", "https://s3batchco1.example.com/1")
    p2, _ = make_pending_posting(db, "S3BatchCo2", "https://s3batchco2.example.com/1")

    with patch.object(stage3_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage3_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_batch = MagicMock(id="msgbatch_s3_123")
            mock_client.messages.batches.create.return_value = mock_batch
            mock_client_cls.return_value = mock_client

            result = stage3_batch.submit_pending_batch(db, posting_ids=[p1.id, p2.id])

    assert result == {
        "batch_id": "msgbatch_s3_123",
        "submitted": 2,
        "capped": False,
        "excluded_unhydrated": 0,
    }
    submitted_requests = mock_client.messages.batches.create.call_args.kwargs["requests"]
    custom_ids = {r["custom_id"] for r in submitted_requests}
    assert custom_ids == {f"posting-{p1.id}", f"posting-{p2.id}"}
    assert submitted_requests[0]["params"]["model"] == stage3_batch.STAGE3_MODEL

    from app.services import batches as batches_service
    from app.services.profile import get_current_profile

    tracked = [
        s for s in batches_service.list_batch_submissions(db) if s.batch_id == "msgbatch_s3_123"
    ]
    assert len(tracked) == 1
    assert tracked[0].kind == "stage3_analysis"
    assert tracked[0].submitted_count == 2
    # Recorded so apply_batch_results can stamp the correct provenance onto
    # verdicts even if results come back after the profile changes again.
    assert tracked[0].profile_version_id == get_current_profile(db).id


def test_submit_pending_batch_excludes_unhydrated_scraped_postings(db):
    eligible, _ = make_pending_posting(db, "S3HydratedCo", "https://s3hydratedco.example.com/1")
    unhydrated = make_unhydrated_scraped_posting(
        db, "S3UnhydratedCo", "https://s3unhydratedco.example.com/1"
    )

    with patch.object(stage3_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage3_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.create.return_value = MagicMock(id="msgbatch_s3_excl")
            mock_client_cls.return_value = mock_client

            result = stage3_batch.submit_pending_batch(db, posting_ids=[eligible.id, unhydrated.id])

    assert result["submitted"] == 1
    assert result["excluded_unhydrated"] == 1
    submitted_requests = mock_client.messages.batches.create.call_args.kwargs["requests"]
    submitted_ids = {r["custom_id"] for r in submitted_requests}
    assert submitted_ids == {f"posting-{eligible.id}"}


def test_submit_pending_batch_returns_none_batch_id_when_nothing_pending(db):
    with patch.object(stage3_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage3_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            result = stage3_batch.submit_pending_batch(db, posting_ids=[999999])

    assert result == {"batch_id": None, "submitted": 0, "capped": False, "excluded_unhydrated": 0}
    mock_client.messages.batches.create.assert_not_called()


def test_submit_pending_batch_include_stale_picks_up_a_stale_result(db):
    # 4.4's roadmap -> profile -> re-fit loop: after a profile edit makes a
    # posting's Stage 3 result stale, include_stale is how the user
    # deliberately re-runs it.
    from app.services.postings import run_stage3_analysis

    posting, verdict = make_pending_posting(db, "S3StaleCo", "https://s3staleco.example.com/1")
    with patch("app.services.postings.stage3_service.analyze") as mock_analyze:
        mock_analyze.return_value = (
            MagicMock(model_dump=lambda mode: {"verdict": "Weak Fit"}),
            100,
        )
        run_stage3_analysis(db, posting, verdict)

    create_profile_version(db, snapshot=EMPTY_SNAPSHOT)  # bumps "current" -- verdict is now stale

    with patch.object(stage3_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage3_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.create.return_value = MagicMock(
                id="msgbatch_s3_stale_default"
            )
            mock_client_cls.return_value = mock_client
            default_result = stage3_batch.submit_pending_batch(db, posting_ids=[posting.id])
    assert default_result["submitted"] == 0  # never included by default

    with patch.object(stage3_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage3_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.create.return_value = MagicMock(
                id="msgbatch_s3_stale_included"
            )
            mock_client_cls.return_value = mock_client
            stale_result = stage3_batch.submit_pending_batch(
                db, posting_ids=[posting.id], include_stale=True
            )
    assert stale_result["submitted"] == 1


def test_submit_pending_batch_raises_stage2_unavailable_without_api_key(db):
    with patch.object(stage3_batch.settings, "anthropic_api_key", None):
        with patch("app.services.stage3_batch.Anthropic") as mock_client_cls:
            with pytest.raises(stage3_batch.Stage2UnavailableError):
                stage3_batch.submit_pending_batch(db)
            mock_client_cls.assert_not_called()


def _fake_succeeded_item(custom_id: str):
    tool_use_block = MagicMock(type="tool_use", input=dict(FAKE_STAGE3_INPUT))
    message = MagicMock(
        content=[tool_use_block], usage=MagicMock(input_tokens=1000, output_tokens=200)
    )
    return MagicMock(custom_id=custom_id, result=MagicMock(type="succeeded", message=message))


def test_apply_batch_results_maps_results_back_to_correct_posting(db):
    p1, _ = make_pending_posting(db, "S3BatchCo3", "https://s3batchco3.example.com/1")
    p2, _ = make_pending_posting(db, "S3BatchCo4", "https://s3batchco4.example.com/1")

    items = [_fake_succeeded_item(f"posting-{p1.id}"), _fake_succeeded_item(f"posting-{p2.id}")]
    with patch.object(stage3_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage3_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.retrieve.return_value = MagicMock(
                processing_status="ended"
            )
            mock_client.messages.batches.results.return_value = items
            mock_client_cls.return_value = mock_client

            result = stage3_batch.apply_batch_results(db, "msgbatch_s3_123")

    assert result["status"] == "ended"
    assert result["applied"] == 2
    assert result["errors"] == 0
    assert result["skipped_stale"] == 0
    assert set(result["posting_ids"]) == {p1.id, p2.id}
    db.refresh(p1.verdicts[0])
    db.refresh(p2.verdicts[0])
    assert p1.verdicts[0].stage3_result["verdict"] == "Strong Fit"
    assert p1.verdicts[0].stage3_tokens_used == 1200
    assert p2.verdicts[0].stage3_result["verdict"] == "Strong Fit"
    # "msgbatch_s3_123" was never recorded via record_batch_submission in
    # this test -- unknown provenance, correctly left null rather than
    # guessed. See test_apply_batch_results_stamps_the_tracked_profile_version.
    assert p1.verdicts[0].stage3_profile_version_id is None


def test_apply_batch_results_stamps_the_tracked_profile_version(db):
    from app.services import batches as batches_service

    p1, _ = make_pending_posting(db, "S3BatchCo3b", "https://s3batchco3b.example.com/1")
    submitted_version = create_profile_version(db, snapshot=EMPTY_SNAPSHOT)
    batches_service.record_batch_submission(
        db,
        "msgbatch_s3_tracked_provenance",
        kind="stage3_analysis",
        submitted_count=1,
        capped=False,
        profile_version_id=submitted_version.id,
    )
    # Profile changes again after submission but before results are applied
    # -- apply must still stamp the version that was CURRENT AT SUBMIT TIME,
    # not whatever's current now.
    create_profile_version(db, snapshot=EMPTY_SNAPSHOT)

    items = [_fake_succeeded_item(f"posting-{p1.id}")]
    with patch.object(stage3_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage3_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.retrieve.return_value = MagicMock(
                processing_status="ended"
            )
            mock_client.messages.batches.results.return_value = items
            mock_client_cls.return_value = mock_client

            stage3_batch.apply_batch_results(db, "msgbatch_s3_tracked_provenance")

    db.refresh(p1.verdicts[0])
    assert p1.verdicts[0].stage3_profile_version_id == submitted_version.id


def test_apply_batch_results_updates_tracked_submission_status(db):
    from app.services import batches as batches_service

    batches_service.record_batch_submission(
        db, "msgbatch_s3_tracked", kind="stage3_analysis", submitted_count=0, capped=False
    )
    with patch.object(stage3_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage3_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.retrieve.return_value = MagicMock(
                processing_status="in_progress"
            )
            mock_client_cls.return_value = mock_client

            stage3_batch.apply_batch_results(db, "msgbatch_s3_tracked")

    tracked = [
        s for s in batches_service.list_batch_submissions(db) if s.batch_id == "msgbatch_s3_tracked"
    ]
    assert tracked[0].last_status == "in_progress"
    assert tracked[0].last_checked_at is not None


def test_apply_batch_results_reports_in_progress_status_without_applying(db):
    with patch.object(stage3_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage3_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.retrieve.return_value = MagicMock(
                processing_status="in_progress"
            )
            mock_client_cls.return_value = mock_client

            result = stage3_batch.apply_batch_results(db, "msgbatch_s3_123")

    assert result == {
        "status": "in_progress",
        "applied": 0,
        "errors": 0,
        "skipped_stale": 0,
        "posting_ids": [],
    }
    mock_client.messages.batches.results.assert_not_called()


def test_apply_batch_results_counts_errored_without_raising(db):
    p1, _ = make_pending_posting(db, "S3BatchCo5", "https://s3batchco5.example.com/1")
    p2, _ = make_pending_posting(db, "S3BatchCo6", "https://s3batchco6.example.com/1")

    errored_item = MagicMock(
        custom_id=f"posting-{p2.id}", result=MagicMock(type="errored", error=MagicMock())
    )
    items = [_fake_succeeded_item(f"posting-{p1.id}"), errored_item]
    with patch.object(stage3_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage3_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.retrieve.return_value = MagicMock(
                processing_status="ended"
            )
            mock_client.messages.batches.results.return_value = items
            mock_client_cls.return_value = mock_client

            result = stage3_batch.apply_batch_results(db, "msgbatch_s3_123")

    assert result["status"] == "ended"
    assert result["applied"] == 1
    assert result["errors"] == 1
    assert result["skipped_stale"] == 0
    assert result["posting_ids"] == [p1.id]


def test_apply_batch_results_skips_stale_when_already_analyzed(db):
    p1, verdict = make_pending_posting(db, "S3BatchCo7", "https://s3batchco7.example.com/1")
    # Simulate the posting having already been analyzed via another path
    # (e.g. a manual "Analyze for fit" click) between submission and result
    # application.
    verdict.stage3_result = dict(FAKE_STAGE3_INPUT)
    db.flush()

    items = [_fake_succeeded_item(f"posting-{p1.id}")]
    with patch.object(stage3_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage3_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.retrieve.return_value = MagicMock(
                processing_status="ended"
            )
            mock_client.messages.batches.results.return_value = items
            mock_client_cls.return_value = mock_client

            result = stage3_batch.apply_batch_results(db, "msgbatch_s3_123")

    assert result == {
        "status": "ended",
        "applied": 0,
        "errors": 0,
        "skipped_stale": 1,
        "posting_ids": [p1.id],
    }


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
    p_relevant, _ = make_pending_posting(
        db,
        "S3RelCo1",
        "https://s3relco1.example.com/1",
        raw_jd_text="Build distributed systems on kubernetes at scale.",
    )
    p_irrelevant, _ = make_pending_posting(
        db,
        "S3RelCo2",
        "https://s3relco2.example.com/1",
        raw_jd_text="Sales representative, enterprise accounts, quota driven.",
    )

    with patch.object(stage3_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage3_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.create.return_value = MagicMock(id="msgbatch_s3_123")
            mock_client_cls.return_value = mock_client

            result = stage3_batch.submit_pending_batch(
                db, posting_ids=[p_relevant.id, p_irrelevant.id], limit=1
            )

    assert result == {
        "batch_id": "msgbatch_s3_123",
        "submitted": 1,
        "capped": True,
        "excluded_unhydrated": 0,
    }
    submitted_requests = mock_client.messages.batches.create.call_args.kwargs["requests"]
    submitted_ids = {r["custom_id"] for r in submitted_requests}
    assert submitted_ids == {f"posting-{p_relevant.id}"}


def test_submit_pending_batch_uncapped_when_limit_is_none(db):
    p1, _ = make_pending_posting(db, "S3RelCo3", "https://s3relco3.example.com/1")
    p2, _ = make_pending_posting(db, "S3RelCo4", "https://s3relco4.example.com/1")

    with patch.object(stage3_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage3_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.create.return_value = MagicMock(id="msgbatch_s3_123")
            mock_client_cls.return_value = mock_client

            result = stage3_batch.submit_pending_batch(db, posting_ids=[p1.id, p2.id], limit=None)

    assert result == {
        "batch_id": "msgbatch_s3_123",
        "submitted": 2,
        "capped": False,
        "excluded_unhydrated": 0,
    }


# -- eligibility: "ready" gate and in-flight exclusion --


def _submit(db, posting_ids, **kw):
    with patch.object(stage3_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage3_batch.Anthropic") as mock_client_cls:
            mock_client_cls.return_value.messages.batches.create.return_value = MagicMock(
                id="msgbatch_gate"
            )
            return stage3_batch.submit_pending_batch(db, posting_ids=posting_ids, **kw)


def test_stage3_skips_postings_stage2_rejected_unless_include_unready(db):
    from tests.test_stage2_batch import make_analyze_posting

    skipped, v = make_analyze_posting(db, "GateCoSkip", "https://gatecoskip.example.com/1")
    v.stage2_result = "skip"
    passed, v2 = make_analyze_posting(db, "GateCoPass", "https://gatecopass.example.com/1")
    v2.stage2_result = "analyze"
    fast, _ = make_pending_posting(
        db, "GateCoFast", "https://gatecofast.example.com/1"
    )  # Stage 1 fast-track
    db.flush()

    ids = [skipped.id, passed.id, fast.id]
    assert {p.id for p in stage3_batch.list_stage3_pending(db, posting_ids=ids)} == {
        passed.id,
        fast.id,
    }
    assert {
        p.id for p in stage3_batch.list_stage3_pending(db, posting_ids=ids, include_unready=True)
    } == set(ids)


def test_stage3_excludes_postings_in_an_unapplied_batch_until_it_is_applied(db):
    from app.services import batches as batches_service

    posting, _ = make_pending_posting(db, "InFlightCo", "https://inflightco.example.com/1")
    assert _submit(db, [posting.id])["submitted"] == 1
    assert stage3_batch.list_stage3_pending(db, posting_ids=[posting.id]) == []
    assert _submit(db, [posting.id])["submitted"] == 0  # would have been a double submission

    batches_service.mark_applied(db, "msgbatch_gate")
    assert [p.id for p in stage3_batch.list_stage3_pending(db, posting_ids=[posting.id])] == [
        posting.id
    ]


def test_apply_marks_the_tracked_batch_applied_only_once_it_has_ended(db):
    from app.services import batches as batches_service

    batches_service.record_batch_submission(
        db,
        "msgbatch_s3_marked",
        kind="stage3_analysis",
        submitted_count=0,
        capped=False,
        posting_ids=[],
    )

    def apply_with_status(status):
        with patch.object(stage3_batch.settings, "anthropic_api_key", "fake-key"):
            with patch("app.services.stage3_batch.Anthropic") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.messages.batches.retrieve.return_value = MagicMock(
                    processing_status=status
                )
                mock_client.messages.batches.results.return_value = []
                mock_client_cls.return_value = mock_client
                stage3_batch.apply_batch_results(db, "msgbatch_s3_marked")
        return batches_service.get_batch_submission(db, "msgbatch_s3_marked").applied_at

    assert apply_with_status("in_progress") is None
    assert apply_with_status("ended") is not None
