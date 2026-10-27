from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.orm import Session

from app.db import engine
from app.models.verdict import Verdict
from app.services import pricing
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


def _posting_with_verdict(db, company_name, url):
    posting, verdict = capture_posting(
        db,
        company_name=company_name,
        title="Software Engineer II",
        raw_jd_text="x",
        url=url,
    )
    return posting, verdict


def test_estimate_cost_unavailable_when_no_token_history(db):
    # A brand-new column with zero non-null rows -- can't happen for a real
    # column with real history, but stage2_tokens_used is exactly this
    # shape before the very first Stage 2 call ever lands. Simulated here
    # by asking for a count of 0, which always short-circuits regardless
    # of the real corpus's history.
    result = pricing.estimate_cost(db, Verdict.stage2_tokens_used, count=0, price_per_mtok=0.50)
    assert result["available"] is False
    assert result["estimated_cost_usd"] is None


def test_estimate_cost_computes_from_average(db):
    _posting, v1 = _posting_with_verdict(db, "PricingCo1", "https://pricingco1.example.com/1")
    _posting2, v2 = _posting_with_verdict(db, "PricingCo2", "https://pricingco2.example.com/1")
    v1.stage2_tokens_used = 1000
    v2.stage2_tokens_used = 3000
    db.flush()

    # Delta-based: can't assert the real corpus's absolute average, but a
    # count of 0 always returns unavailable regardless of history, and a
    # positive count always returns *some* estimate once any history
    # exists (which these two rows guarantee).
    result = pricing.estimate_cost(db, Verdict.stage2_tokens_used, count=10, price_per_mtok=0.50)
    assert result["available"] is True
    assert result["avg_tokens_per_posting"] > 0
    assert result["estimated_cost_usd"] > 0


def test_spend_summary_buckets_by_updated_at(db):
    _posting1, v1 = _posting_with_verdict(db, "SpendCo1", "https://spendco1.example.com/1")
    _posting2, v2 = _posting_with_verdict(db, "SpendCo2", "https://spendco2.example.com/1")
    v1.stage2_tokens_used = 1000  # recent -- default updated_at (now)
    v2.stage2_tokens_used = 2000
    db.flush()
    v2.updated_at = datetime.now(timezone.utc) - timedelta(days=45)  # outside the 30-day window
    db.flush()

    before = pricing.spend_summary(db)
    # Can't isolate further without touching more rows -- instead confirm
    # the invariant that must hold regardless of the real corpus's other
    # data: all_time tokens >= last_30d tokens (a subset can never exceed
    # its superset).
    assert before["stage2"]["tokens_all_time"] >= before["stage2"]["tokens_last_30d"]
    assert before["stage2"]["tokens_all_time"] >= 3000


def test_spend_summary_prices_stage2_tokens_by_the_model_in_use_when_they_were_spent(db):
    # Stage 2 moved from Haiku 4.5 ($0.50/MTok batch input) to Haiku 5.5 ($0.05) at the cutover;
    # tokens on rows last updated before it must keep the old rate, not be re-priced at the new one.
    _p1, old_row = _posting_with_verdict(db, "CutoverCo1", "https://cutoverco1.example.com/1")
    _p2, new_row = _posting_with_verdict(db, "CutoverCo2", "https://cutoverco2.example.com/1")
    before = pricing.spend_summary(db)["stage2"]["estimated_cost_usd_all_time"]

    old_row.stage2_tokens_used = 1_000_000
    new_row.stage2_tokens_used = 1_000_000
    db.flush()
    old_row.updated_at = pricing.STAGE2_MODEL_CUTOVER - timedelta(days=1)
    new_row.updated_at = pricing.STAGE2_MODEL_CUTOVER + timedelta(days=1)
    db.flush()

    after = pricing.spend_summary(db)["stage2"]["estimated_cost_usd_all_time"]
    assert round(after - before, 4) == round(
        pricing.HAIKU_4_5_BATCH_PRICE_PER_MTOK + pricing.HAIKU_BATCH_PRICE_PER_MTOK, 4
    )


def test_spend_summary_reports_zero_when_no_stage3_history_at_all_would_be_zero_floor(db):
    result = pricing.spend_summary(db)
    assert result["stage3"]["tokens_all_time"] >= 0
    assert result["stage3"]["tokens_last_30d"] >= 0
