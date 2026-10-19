"""build_verdict_out is a pure function of (Verdict, current_profile_version_id)
-- no DB needed, a plain in-memory Verdict instance is enough."""

from app.models.verdict import Verdict
from app.schemas.verdict import build_verdict_out


def _verdict(**overrides) -> Verdict:
    defaults = dict(
        id=1,
        posting_id=1,
        profile_version_id=1,
        stage1_result="analyze",
        stage1_reason=None,
        stage2_result=None,
        stage2_reason=None,
        stage2_tokens_used=None,
        stage2_profile_version_id=None,
        stage3_triggered_at=None,
        stage3_result=None,
        stage3_tokens_used=None,
        stage3_profile_version_id=None,
    )
    defaults.update(overrides)
    v = Verdict(**defaults)
    # created_at/updated_at are server_default -- never populated on an
    # unflushed instance, but VerdictOut requires them.
    from datetime import datetime, timezone

    v.created_at = datetime.now(timezone.utc)
    v.updated_at = datetime.now(timezone.utc)
    return v


def test_no_result_is_never_stale():
    v = _verdict(stage2_result=None, stage3_result=None)
    out = build_verdict_out(v, current_profile_version_id=5)
    assert out.stage2_stale is False
    assert out.stage3_stale is False


def test_result_against_current_version_is_not_stale():
    v = _verdict(stage2_result="skip", stage2_profile_version_id=5)
    out = build_verdict_out(v, current_profile_version_id=5)
    assert out.stage2_stale is False


def test_result_against_an_older_version_is_stale():
    v = _verdict(stage2_result="skip", stage2_profile_version_id=5)
    out = build_verdict_out(v, current_profile_version_id=6)
    assert out.stage2_stale is True


def test_stage3_result_against_an_older_version_is_stale():
    v = _verdict(stage3_result={"verdict": "Strong Fit"}, stage3_profile_version_id=5)
    out = build_verdict_out(v, current_profile_version_id=6)
    assert out.stage3_stale is True


def test_null_provenance_is_never_stale_even_with_a_result():
    # Legacy rows written before stage{2,3}_profile_version_id existed --
    # unknown provenance must never be reported stale (would be guessing).
    v = _verdict(stage2_result="skip", stage2_profile_version_id=None)
    out = build_verdict_out(v, current_profile_version_id=99)
    assert out.stage2_stale is False

    v3 = _verdict(stage3_result={"verdict": "Skip"}, stage3_profile_version_id=None)
    out3 = build_verdict_out(v3, current_profile_version_id=99)
    assert out3.stage3_stale is False


def test_is_stale_is_no_longer_exposed():
    v = _verdict()
    out = build_verdict_out(v, current_profile_version_id=1)
    assert not hasattr(out, "is_stale")
