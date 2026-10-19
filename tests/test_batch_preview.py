import pytest
from sqlalchemy.orm import Session

from app.db import engine
from app.services import batch_preview
from app.services.profile import create_profile_version
from tests.test_stage2_batch import make_analyze_posting, make_unhydrated_scraped_posting

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


def test_preview_stage2_counts_eligible_postings(db):
    p1, _ = make_analyze_posting(db, "PreviewCo1", "https://previewco1.example.com/1")
    p2, _ = make_analyze_posting(db, "PreviewCo2", "https://previewco2.example.com/1")

    result = batch_preview.preview_stage2(db, posting_ids=[p1.id, p2.id], limit=30)
    assert result["eligible"] == 2
    assert result["would_submit"] == 2
    assert result["capped"] is False
    assert result["excluded_unhydrated"] == 0


def test_preview_stage2_excludes_unhydrated_scraped_postings(db):
    hydrated, _ = make_analyze_posting(
        db, "PreviewHydratedCo", "https://previewhydratedco.example.com/1"
    )
    unhydrated = make_unhydrated_scraped_posting(
        db, "PreviewUnhydratedCo", "https://previewunhydratedco.example.com/1"
    )

    result = batch_preview.preview_stage2(db, posting_ids=[hydrated.id, unhydrated.id], limit=30)
    assert result["eligible"] == 1
    assert result["would_submit"] == 1
    assert result["excluded_unhydrated"] == 1


def test_preview_stage2_respects_cap(db):
    p1, _ = make_analyze_posting(db, "CapCo1", "https://capco1.example.com/1")
    p2, _ = make_analyze_posting(db, "CapCo2", "https://capco2.example.com/1")
    p3, _ = make_analyze_posting(db, "CapCo3", "https://capco3.example.com/1")

    result = batch_preview.preview_stage2(db, posting_ids=[p1.id, p2.id, p3.id], limit=2)
    assert result["eligible"] == 3
    assert result["would_submit"] == 2
    assert result["capped"] is True


def test_preview_stage2_matches_submit_pending_batch_semantics(db):
    """Cross-check against the real submit path's own counts on the same
    fixture data -- preview must never diverge from what submitting would
    actually do."""
    from unittest.mock import MagicMock, patch

    from app.services import stage2_batch

    p1, _ = make_analyze_posting(db, "CrossCheckCo1", "https://crosscheckco1.example.com/1")
    p2, _ = make_analyze_posting(db, "CrossCheckCo2", "https://crosscheckco2.example.com/1")

    preview = batch_preview.preview_stage2(db, posting_ids=[p1.id, p2.id], limit=30)

    with patch.object(stage2_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage2_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.create.return_value = MagicMock(id="msgbatch_crosscheck")
            mock_client_cls.return_value = mock_client
            submitted = stage2_batch.submit_pending_batch(db, posting_ids=[p1.id, p2.id])

    assert preview["would_submit"] == submitted["submitted"]
    assert preview["capped"] == submitted["capped"]
    assert preview["excluded_unhydrated"] == submitted["excluded_unhydrated"]


def test_preview_stage2_available_false_without_absolute_zero_hack(db):
    p1, _ = make_analyze_posting(db, "NoTokenCo", "https://notokenco.example.com/1")
    result = batch_preview.preview_stage2(db, posting_ids=[p1.id], limit=0)
    # limit=0 forces would_submit to 0 regardless of history, which always
    # reports unavailable (see pricing.estimate_cost) -- a reliable way to
    # exercise the "no cost" branch without depending on the real corpus
    # ever having zero Stage 2 token history.
    assert result["would_submit"] == 0
    assert result["available"] is False


def test_preview_stage3_counts_eligible_postings(db):
    p1, v1 = make_analyze_posting(db, "PreviewStage3Co1", "https://previewstage3co1.example.com/1")
    v1.stage2_result = "analyze"  # Stage 2 passed it, so it is "ready" for Stage 3
    db.flush()

    result = batch_preview.preview_stage3(db, posting_ids=[p1.id], limit=30)
    assert result["eligible"] == 1
    assert result["would_submit"] == 1
    assert result["capped"] is False


def test_preview_stage3_include_stale_agrees_with_submit(db):
    # Cross-check against the real submit path's own counts on the same
    # fixture data, same reasoning as
    # test_preview_stage2_matches_submit_pending_batch_semantics.
    from unittest.mock import MagicMock, patch

    from app.services import stage3_batch
    from app.services.postings import capture_posting, run_stage3_analysis

    posting, verdict = capture_posting(
        db,
        company_name="PreviewStage3Stale",
        title="ML Engineer",
        raw_jd_text="x",
        url="https://previewstage3stale.example.com/1",
    )
    with patch("app.services.postings.stage3_service.analyze") as mock_analyze:
        mock_analyze.return_value = (
            MagicMock(model_dump=lambda mode: {"verdict": "Weak Fit"}),
            100,
        )
        run_stage3_analysis(db, posting, verdict)
    create_profile_version(db, snapshot=EMPTY_SNAPSHOT)  # bumps "current" -- verdict is now stale

    preview_default = batch_preview.preview_stage3(db, posting_ids=[posting.id], limit=30)
    preview_stale = batch_preview.preview_stage3(
        db, posting_ids=[posting.id], limit=30, include_stale=True
    )
    assert preview_default["would_submit"] == 0
    assert preview_stale["would_submit"] == 1

    with patch.object(stage3_batch.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage3_batch.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.batches.create.return_value = MagicMock(
                id="msgbatch_preview_stale"
            )
            mock_client_cls.return_value = mock_client
            submitted = stage3_batch.submit_pending_batch(
                db, posting_ids=[posting.id], include_stale=True
            )
    assert preview_stale["would_submit"] == submitted["submitted"]
