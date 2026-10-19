"""Batch API cost estimation. Stage 2 uses Haiku 4.5 (see
app.services.stage2.STAGE2_MODEL), Stage 3 uses Sonnet 5 (see
app.services.stage3.STAGE3_MODEL). Prices are per-million-tokens, batch
rate (50% off standard), taken from Anthropic's published pricing as of 2026-09.
Update the two constants below if pricing changes -- nothing else in the
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

HAIKU_BATCH_PRICE_PER_MTOK = 0.50
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


def _stage_totals(db: Session, column, since: datetime | None = None) -> dict:
    stmt = select(func.coalesce(func.sum(column), 0)).where(column.is_not(None))
    if since is not None:
        stmt = stmt.where(Verdict.updated_at >= since)
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

    def stage_summary(column, price_per_mtok):
        all_time = _stage_totals(db, column)
        last_30d = _stage_totals(db, column, since=since)
        return {
            "tokens_all_time": all_time["tokens"],
            "estimated_cost_usd_all_time": round(
                all_time["tokens"] / 1_000_000 * price_per_mtok, 4
            ),
            "tokens_last_30d": last_30d["tokens"],
            "estimated_cost_usd_last_30d": round(
                last_30d["tokens"] / 1_000_000 * price_per_mtok, 4
            ),
        }

    return {
        "stage2": stage_summary(Verdict.stage2_tokens_used, HAIKU_BATCH_PRICE_PER_MTOK),
        "stage3": stage_summary(Verdict.stage3_tokens_used, SONNET_BATCH_PRICE_PER_MTOK),
    }
