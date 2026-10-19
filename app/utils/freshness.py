from datetime import date, datetime


def freshness_flag(
    posted_date: date | None,
    first_seen_at: datetime | None,
    closed_at: datetime | None = None,
    today: date | None = None,
) -> str:
    """Computed at query time, never stored -- spec section 6.

    closed_at wins over everything else: a closed posting is "closed"
    regardless of how recently it was posted or last seen. Otherwise based
    on posted_date, falling back to first_seen_at only when posted_date is
    unknown (e.g. a scrape that couldn't parse a date string).

    Not last_seen_at: once repost_count only tracks genuine reopen events
    (see upsert_posting) rather than every re-sighting on an unchanged
    still-open posting, last_seen_at stops being a meaningful proxy for
    "how fresh is this listing" -- a posting seen again today because it's
    simply still open isn't any fresher than it was yesterday.
    """
    if closed_at is not None:
        return "closed"

    today = today or date.today()
    reference = posted_date or (first_seen_at.date() if first_seen_at else today)

    days_old = (today - reference).days
    if days_old < 30:
        return "fresh"
    elif days_old <= 45:
        return "aging"
    else:
        return "stale"
