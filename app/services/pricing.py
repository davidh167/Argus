"""Batch API cost estimation. Stage 2 uses Haiku 5.5 (see
app.services.stage2.STAGE2_MODEL; Haiku 4.5 before STAGE2_MODEL_CUTOVER), Stage 3 uses Sonnet 5 (see
app.services.stage3.STAGE3_MODEL). Prices are per-million-tokens, batch
rate (50% off standard), taken from Anthropic's published pricing as of 2026-10.
Update the constants below if pricing changes -- nothing else in the
codebase hardcodes a rate.

Verdict.stage2_tokens_used/stage3_tokens_used combine input+output token
counts (no separate breakdown is stored anywhere), so there's no way to
apply the input/output rates separately per posting. Using each model's
batch INPUT rate as the blended estimate is a defensible approximation:
input (the JD text, ~1-2k tokens) is the overwhelming majority of every
request; output is a short structured JSON verdict. Every number this
module returns is explicitly an estimate, never a bill line -- callers
must surface that.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.verdict import Verdict

# Haiku 5.5: $0.10/MTok standard input for prompts up to 100K tokens (a posting is a few thousand), half
# that in a batch. Haiku 4.5 was $1.00 standard, $0.50 batch.
HAIKU_BATCH_PRICE_PER_MTOK = 0.05
HAIKU_4_5_BATCH_PRICE_PER_MTOK = 0.50
# Stage 2 tokens on verdicts last updated before this were spent on Haiku 4.5. Verdict.updated_at moves
# when Stage 3 is added later, so an old row can look newer than it is (see spend_summary).
STAGE2_MODEL_CUTOVER = datetime(2026, 10, 26, tzinfo=timezone.utc)
SONNET_BATCH_PRICE_PER_MTOK = 1.00
# Synchronous (non-batch) Sonnet 5 rates, per MTok -- used by calls that can't wait for a batch.
SONNET_SYNC_INPUT_PRICE_PER_MTOK = 2.00
SONNET_SYNC_OUTPUT_PRICE_PER_MTOK = 10.00

SPEND_WINDOW_DAYS = 30


def _average_tokens(db: Session, column) -> float | None:
    avg = db.scalar(select(func.avg(column)).where(column.is_not(None)))
    return float(avg) if avg is not None else None


def estimate_cost(db: Session, column, count: int, price_per_mtok: float) -> dict:
    """count is the number of postings that WOULD be submitted -- the
    caller (batch_preview) supplies it; this function only turns a count
    into a cost given the historical per-posting average. available=False
    (no fabricated number) when there's no token history yet for that
    stage -- e.g. before the very first Stage 2/3 call has ever landed."""
    avg_tokens = _average_tokens(db, column)
    if avg_tokens is None or count == 0:
        return {"available": False, "avg_tokens_per_posting": None, "estimated_cost_usd": None}
    estimated_cost = (avg_tokens * count) / 1_000_000 * price_per_mtok
    return {
        "available": True,
        "avg_tokens_per_posting": round(avg_tokens),
        "estimated_cost_usd": round(estimated_cost, 4),
    }


def _stage_totals(
    db: Session, column, since: datetime | None = None, until: datetime | None = None
) -> dict:
    stmt = select(func.coalesce(func.sum(column), 0)).where(column.is_not(None))
    if since is not None:
        stmt = stmt.where(Verdict.updated_at >= since)
    if until is not None:
        stmt = stmt.where(Verdict.updated_at < until)
    total_tokens = db.scalar(stmt) or 0
    return {"tokens": int(total_tokens)}


def spend_summary(db: Session) -> dict:
    """Tokens + estimated $ spent so far on Stage 2/Stage 3, each broken
    into last-30-days and all-time. Buckets by Verdict.updated_at, not
    created_at -- created_at is set at Stage 1 (posting ingestion), which
    can predate a Stage 2/3 result by weeks. Known caveat: if a verdict
    row is touched again later (e.g. Stage 3 added on top of an old Stage
    2 row), that row's updated_at moves forward, making Stage 2's
    historical tokens on that row look more recently spent than they
    actually were. Disclosed here rather than hidden -- there's no
    per-stage timestamp stored to do better."""
    since = datetime.now(timezone.utc) - timedelta(days=SPEND_WINDOW_DAYS)

    def cost(column, since, segments):
        """segments: [(price_per_mtok, start, end)] over Verdict.updated_at, start inclusive and end
        exclusive (None = unbounded). Each row falls in exactly one segment."""
        usd = 0.0
        for price_per_mtok, start, end in segments:
            lower = max((d for d in (start, since) if d is not None), default=None)
            tokens = _stage_totals(db, column, since=lower, until=end)["tokens"]
            usd += tokens / 1_000_000 * price_per_mtok
        return round(usd, 4)

    def stage_summary(column, segments):
        return {
            "tokens_all_time": _stage_totals(db, column)["tokens"],
            "estimated_cost_usd_all_time": cost(column, None, segments),
            "tokens_last_30d": _stage_totals(db, column, since=since)["tokens"],
            "estimated_cost_usd_last_30d": cost(column, since, segments),
        }

    return {
        "stage2": stage_summary(
            Verdict.stage2_tokens_used,
            [
                (HAIKU_4_5_BATCH_PRICE_PER_MTOK, None, STAGE2_MODEL_CUTOVER),
                (HAIKU_BATCH_PRICE_PER_MTOK, STAGE2_MODEL_CUTOVER, None),
            ],
        ),
        "stage3": stage_summary(
            Verdict.stage3_tokens_used, [(SONNET_BATCH_PRICE_PER_MTOK, None, None)]
        ),
    }
