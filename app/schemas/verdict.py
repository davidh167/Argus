from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from app.models.verdict import Verdict


class VerdictOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    posting_id: int
    profile_version_id: int
    created_at: datetime
    updated_at: datetime

    stage1_result: str | None
    stage1_reason: str | None

    stage2_result: str | None
    stage2_reason: str | None
    stage2_tokens_used: int | None
    stage2_stale: bool

    stage3_triggered_at: datetime | None
    stage3_result: dict | None
    stage3_tokens_used: int | None
    stage3_stale: bool


def build_verdict_out(verdict: "Verdict", current_profile_version_id: int) -> VerdictOut:
    """The only correct way to build a VerdictOut -- staleness is computed
    here at read time (same philosophy as freshness_flag), never stored.
    A result is stale when the profile version it was actually computed
    against differs from the current one.

    Null provenance (stage{2,3}_profile_version_id is None -- legacy rows
    from before those columns existed, or an untracked Stage 3 batch_id)
    means "unknown," and is never reported stale: guessing would be exactly
    the kind of speculative behavior this codebase avoids. A result with no
    result at all (stage2_result/stage3_result is None) is simply "not run
    yet," not stale.
    """
    stage2_stale = (
        verdict.stage2_result is not None
        and verdict.stage2_profile_version_id is not None
        and verdict.stage2_profile_version_id != current_profile_version_id
    )
    stage3_stale = (
        verdict.stage3_result is not None
        and verdict.stage3_profile_version_id is not None
        and verdict.stage3_profile_version_id != current_profile_version_id
    )
    return VerdictOut(
        id=verdict.id,
        posting_id=verdict.posting_id,
        profile_version_id=verdict.profile_version_id,
        created_at=verdict.created_at,
        updated_at=verdict.updated_at,
        stage1_result=verdict.stage1_result,
        stage1_reason=verdict.stage1_reason,
        stage2_result=verdict.stage2_result,
        stage2_reason=verdict.stage2_reason,
        stage2_tokens_used=verdict.stage2_tokens_used,
        stage2_stale=stage2_stale,
        stage3_triggered_at=verdict.stage3_triggered_at,
        stage3_result=verdict.stage3_result,
        stage3_tokens_used=verdict.stage3_tokens_used,
        stage3_stale=stage3_stale,
    )
