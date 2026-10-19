from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

DemandStatus = Literal["have", "developing", "gap"]

RoadmapStatus = Literal["flagged", "researching", "in_progress", "complete"]
RoadmapTriggerType = Literal["job_market_gap", "project_strengthener"]


class SkillTrend(BaseModel):
    """Ephemeral result of detect_skill_trends -- never persisted on its own."""

    skill_or_language: str
    posting_ids: list[int]
    occurrence_count: int


class ReadinessEntry(BaseModel):
    """One posting in one bucket of readiness_view (4.2) -- ephemeral,
    like SkillTrend, recomputed on every call."""

    posting_id: int
    title: str
    company_name: str
    verdict: str
    one_liner: str
    hard_gaps: list[str]
    # Only ever set for a build_toward entry -- the existing SkillRoadmap
    # row (and the specific hard_gaps sentence) this posting's Weak Fit
    # gap matched against.
    roadmap_item_id: int | None = None
    matched_gap: str | None = None


class ReadinessView(BaseModel):
    apply_now: list[ReadinessEntry]
    solid_match: list[ReadinessEntry]
    stretch: list[ReadinessEntry]
    build_toward: list[ReadinessEntry]


class DemandCluster(BaseModel):
    """Ephemeral result of market_demand (4.3) -- never persisted on its
    own, recomputed on every call, same as SkillTrend."""

    skill_or_language: str
    posting_count: int
    posting_ids: list[int]
    status: DemandStatus
    # Sum of the interest weights (roadmap.interest_weights) of this
    # cluster's postings, and how many of them are in the explicit
    # interest set (manual capture / starred / actively applied to).
    interest_weight: float = 0.0
    interest_posting_count: int = 0
    # Distinct phrasings merged into this cluster (up to 10), for display and
    # as context for the skill judgment.
    phrasings: list[str] = []
    # True when status comes from a cached LLM judgment of the current
    # profile (skill_judgment.py) rather than the embedding-similarity guess.
    judged: bool = False
    judgment_reason: str | None = None


class SkillRoadmapOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    skill_or_language: str
    trigger_type: str | None
    is_new_territory: bool | None
    trend_evidence: list[int] | None
    status: str
    what_it_is: str | None
    why_it_matters: str | None
    suggested_scope: str | None
    suggested_stack: str | None
    strengthens_project: str | None
    project_note: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


class GenerateRoadmapBriefRequest(BaseModel):
    skill_or_language: str
    trigger_type: RoadmapTriggerType
    trend_evidence: list[int] = []


class UpdateRoadmapStatusRequest(BaseModel):
    status: RoadmapStatus


class SkillJudgmentPreview(BaseModel):
    to_judge: int
    already_judged: int
    estimated_input_tokens: int
    estimated_cost_usd: float


class SkillJudgmentRun(BaseModel):
    judged: int
    remaining: int
