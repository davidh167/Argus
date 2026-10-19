from datetime import date, datetime

from pydantic import BaseModel

from app.schemas.verdict import VerdictOut


class CapturePostingRequest(BaseModel):
    url: str | None = None
    company_name: str
    title: str
    raw_jd_text: str
    posted_date: date | None = None
    # Free text, not geocoded -- the user is looking at the posting and can
    # type/paste what it says. Never LLM-extracted.
    location: str | None = None


class PostingSummary(BaseModel):
    id: int
    company_id: int
    company_name: str
    title: str
    url: str | None
    source: str | None
    starred: bool
    starred_at: datetime | None
    dismissed: bool = False
    dismissed_at: datetime | None = None
    note: str | None
    freshness: str
    repost_count: int
    stage1_result: str | None = None
    # From the latest verdict, mirroring stage1_result -- lets the desktop
    # list view show a fit badge (postings.fit_tier_case's tiers, derived
    # client-side) without a per-card detail fetch.
    stage2_result: str | None = None
    stage3_verdict: str | None = None
    locations: list[str] | None = None
    workplace_type: str | None = None
    comp_min: int | None = None
    comp_max: int | None = None
    comp_currency: str | None = None
    comp_period: str | None = None
    relevance_score: float | None = None


class JDSectionsOut(BaseModel):
    summary: str
    responsibilities: str
    required_qualifications: str
    preferred_qualifications: str


class PostingDetail(PostingSummary):
    raw_jd_text: str | None
    posted_date: date | None
    first_seen_at: datetime
    last_seen_at: datetime
    closed_at: datetime | None = None
    department: str | None = None
    comp_source: str | None = None
    comp_raw: str | None = None
    detail_fetched_at: datetime | None = None
    # The terms contributing most to relevance_score -- cheap "why is this
    # ranked here" explainability (see relevance.rescore_all).
    relevance_matched_terms: list[str] | None = None
    # Computed at read time from raw_jd_text (app.services.jd_sections),
    # uncapped -- not stored, cheap regex, matches how freshness is also
    # computed at query time rather than persisted.
    jd_sections: JDSectionsOut | None = None
    latest_verdict: VerdictOut | None = None


class PostingCreated(BaseModel):
    posting: PostingDetail
    verdict: VerdictOut


class TriagePendingSubmitted(BaseModel):
    batch_id: str | None
    submitted: int
    capped: bool
    # Pending postings left out because they're source='scraped' with no
    # detail_fetched_at yet -- still just a short listing blurb, not worth
    # the call. Run POST /postings/hydrate, then resubmit.
    excluded_unhydrated: int = 0


class TriagePendingApplied(BaseModel):
    status: str
    applied: int
    errors: int
    skipped_stale: int
    posting_ids: list[int] = []


class AnalyzePendingSubmitted(BaseModel):
    batch_id: str | None
    submitted: int
    capped: bool
    excluded_unhydrated: int = 0


class AnalyzePendingApplied(BaseModel):
    status: str
    applied: int
    errors: int
    skipped_stale: int
    posting_ids: list[int] = []


class HydrateResult(BaseModel):
    hydrated: int
    failed: int
    remaining: int


class RescoreResult(BaseModel):
    rescored: int
    profile_version_id: int


class BatchPreview(BaseModel):
    eligible: int
    would_submit: int
    capped: bool
    excluded_unhydrated: int
    # Postings that would otherwise be eligible but are already in a submitted,
    # not-yet-applied batch of this stage.
    in_pending_batches: int = 0
    # available=False (avg_tokens/estimated_cost_usd both null) when there's
    # no token history yet for this stage -- never a fabricated $0.
    available: bool
    avg_tokens_per_posting: int | None = None
    estimated_cost_usd: float | None = None


class StageSpend(BaseModel):
    tokens_all_time: int
    estimated_cost_usd_all_time: float
    tokens_last_30d: int
    estimated_cost_usd_last_30d: float


class SpendSummary(BaseModel):
    stage2: StageSpend
    stage3: StageSpend


class SimilarPosting(BaseModel):
    posting: PostingSummary
    similarity: float
    # True only for a same-company pair above
    # relevance.SIMILAR_ROLE_SAME_COMPANY_THRESHOLD -- see that constant's
    # docstring for real calibration data and its known limitations.
    # Purely informational: different postings are always different
    # roles, this never merges or dedupes anything.
    possibly_same_role: bool
