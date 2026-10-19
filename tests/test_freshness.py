from datetime import date, datetime, timezone

from app.utils.freshness import freshness_flag

TODAY = date(2026, 9, 24)


def test_closed_wins_over_everything_else():
    # Posted yesterday, but closed -- must still read "closed," not "fresh".
    assert (
        freshness_flag(date(2026, 9, 23), None, closed_at=datetime.now(timezone.utc), today=TODAY)
        == "closed"
    )


def test_fresh_under_30_days():
    assert freshness_flag(date(2026, 9, 1), None, today=TODAY) == "fresh"


def test_aging_between_30_and_45_days():
    assert freshness_flag(date(2026, 8, 10), None, today=TODAY) == "aging"  # 45 days


def test_stale_over_45_days():
    assert freshness_flag(date(2026, 8, 1), None, today=TODAY) == "stale"


def test_falls_back_to_first_seen_at_when_posted_date_is_none():
    recent_first_seen = datetime(2026, 9, 20, tzinfo=timezone.utc)
    assert freshness_flag(None, recent_first_seen, today=TODAY) == "fresh"

    old_first_seen = datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert freshness_flag(None, old_first_seen, today=TODAY) == "stale"


def test_defaults_to_today_when_both_dates_are_unknown():
    # No posted_date, no first_seen_at (shouldn't happen in practice --
    # first_seen_at is server_default=now() -- but must not crash).
    assert freshness_flag(None, None, today=TODAY) == "fresh"


def test_posted_date_takes_priority_over_first_seen_at():
    old_posted = date(2026, 7, 1)  # stale on its own
    recent_first_seen = datetime(2026, 9, 20, tzinfo=timezone.utc)  # fresh on its own
    assert freshness_flag(old_posted, recent_first_seen, today=TODAY) == "stale"
