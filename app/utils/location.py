from app.schemas.profile import LocationPrefs


def _any_contains(locations: list[str] | None, needles: list[str]) -> bool:
    if not locations or not needles:
        return False
    locations_lower = [loc.lower() for loc in locations]
    return any(needle.lower() in loc for needle in needles for loc in locations_lower)


def matches_location(
    prefs: LocationPrefs,
    *,
    workplace_type: str | None,
    locations: list[str] | None,
) -> bool | None:
    """Computed at query time, never stored -- same philosophy as
    freshness_flag. None when the posting has neither workplace_type nor
    locations at all: postings with no location data are neither matched
    nor hidden, they get a "location unknown" state the caller can show
    separately from an actual non-match.

    True when: workplace_type is remote and remote_ok, and (no country
    constraint, or a location matches one of the preferred countries) --
    OR any location contains a preferred city, subject to hybrid_ok/
    onsite_ok gating that city (a hybrid or onsite posting whose city
    matches still doesn't count if that workplace type is turned off).
    The hybrid/onsite flags never gate the remote-country check, and never
    gate a city match on a posting whose workplace_type is remote or
    unknown -- they only apply when the posting IS actually hybrid/onsite.

    exclude_cities/exclude_countries are checked FIRST and unconditionally
    override any positive match -- a hard "no," not subject to hybrid_ok/
    onsite_ok gating. Still a no-op when there's no location text to check
    against, same as every other field here.
    """
    if workplace_type is None and not locations:
        return None

    if _any_contains(locations, prefs.exclude_cities) or _any_contains(
        locations, prefs.exclude_countries
    ):
        return False

    if workplace_type == "remote" and prefs.remote_ok:
        if not prefs.countries or _any_contains(locations, prefs.countries):
            return True

    if prefs.cities and _any_contains(locations, prefs.cities):
        if workplace_type == "hybrid" and not prefs.hybrid_ok:
            pass
        elif workplace_type == "onsite" and not prefs.onsite_ok:
            pass
        else:
            return True

    return False
