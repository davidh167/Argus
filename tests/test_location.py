from app.schemas.profile import Criteria, LocationPrefs
from app.utils.location import matches_location


def test_none_when_no_location_data_at_all():
    prefs = LocationPrefs(cities=["San Francisco"])
    assert matches_location(prefs, workplace_type=None, locations=None) is None
    assert matches_location(prefs, workplace_type=None, locations=[]) is None


def test_remote_matches_when_remote_ok_and_no_country_constraint():
    prefs = LocationPrefs(remote_ok=True)
    assert matches_location(prefs, workplace_type="remote", locations=["United States"]) is True


def test_remote_does_not_match_when_remote_ok_false():
    prefs = LocationPrefs(remote_ok=False)
    assert matches_location(prefs, workplace_type="remote", locations=["United States"]) is False


def test_remote_with_country_constraint_matching():
    prefs = LocationPrefs(remote_ok=True, countries=["United States"])
    assert (
        matches_location(prefs, workplace_type="remote", locations=["Remote, United States"])
        is True
    )


def test_remote_with_country_constraint_not_matching_falls_through():
    prefs = LocationPrefs(remote_ok=True, countries=["Canada"])
    # Remote, but not in a preferred country, and no city preference either.
    assert (
        matches_location(prefs, workplace_type="remote", locations=["Remote, United States"])
        is False
    )


def test_city_match_on_onsite_when_onsite_ok():
    prefs = LocationPrefs(cities=["San Francisco"], onsite_ok=True)
    assert matches_location(prefs, workplace_type="onsite", locations=["San Francisco, CA"]) is True


def test_city_match_gated_off_when_onsite_not_ok():
    prefs = LocationPrefs(cities=["San Francisco"], onsite_ok=False)
    assert (
        matches_location(prefs, workplace_type="onsite", locations=["San Francisco, CA"]) is False
    )


def test_city_match_on_hybrid_when_hybrid_ok():
    prefs = LocationPrefs(cities=["Seattle"], hybrid_ok=True)
    assert matches_location(prefs, workplace_type="hybrid", locations=["Seattle, WA"]) is True


def test_city_match_gated_off_when_hybrid_not_ok():
    prefs = LocationPrefs(cities=["Seattle"], hybrid_ok=False)
    assert matches_location(prefs, workplace_type="hybrid", locations=["Seattle, WA"]) is False


def test_city_match_on_unknown_workplace_type_not_gated():
    # No hybrid/onsite gating applies when workplace_type itself is
    # unknown -- both flags could be False and this must still match.
    prefs = LocationPrefs(cities=["Austin"], hybrid_ok=False, onsite_ok=False)
    assert matches_location(prefs, workplace_type=None, locations=["Austin, TX"]) is True


def test_no_match_when_neither_remote_nor_city_condition_holds():
    prefs = LocationPrefs(cities=["Seattle"])
    assert (
        matches_location(prefs, workplace_type="onsite", locations=["San Francisco, CA"]) is False
    )


def test_default_prefs_do_not_match_a_city_with_no_city_constraint_set():
    # Empty cities list imposes no constraint -- it must not be read as
    # "matches every city."
    prefs = LocationPrefs()
    assert (
        matches_location(prefs, workplace_type="onsite", locations=["San Francisco, CA"]) is False
    )


def test_default_prefs_match_remote_since_remote_ok_defaults_true():
    prefs = LocationPrefs()
    assert matches_location(prefs, workplace_type="remote", locations=["Remote"]) is True


def test_case_insensitive_city_match():
    prefs = LocationPrefs(cities=["san francisco"])
    assert matches_location(prefs, workplace_type="onsite", locations=["SAN FRANCISCO, CA"]) is True


# -- Backward compatibility with the pre-existing free-form shape --


def test_criteria_validates_old_free_form_location_prefs_shape():
    # The real, currently-stored shape (based_in/open_to/hard_no) predates
    # LocationPrefs -- it must still validate (Pydantic ignores unknown
    # keys by default here), falling back to all-permissive defaults
    # rather than raising, since profile reads happen on every request.
    criteria = Criteria.model_validate(
        {
            "skip_patterns": [],
            "target_signals": [],
            "location_prefs": {
                "based_in": "Los Angeles",
                "open_to": ["SF Bay Area", "Remote"],
                "hard_no": ["New York"],
            },
            "seniority_floor": None,
            "seniority_ceiling": None,
        }
    )
    assert criteria.location_prefs == LocationPrefs()  # old keys ignored, defaults used


def test_criteria_validates_empty_dict_location_prefs():
    criteria = Criteria.model_validate(
        {
            "skip_patterns": [],
            "target_signals": [],
            "location_prefs": {},
            "seniority_floor": None,
            "seniority_ceiling": None,
        }
    )
    assert criteria.location_prefs == LocationPrefs()


def test_criteria_validates_new_shape_location_prefs():
    criteria = Criteria.model_validate(
        {
            "skip_patterns": [],
            "target_signals": [],
            "location_prefs": {"cities": ["Los Angeles"], "remote_ok": True, "onsite_ok": False},
            "seniority_floor": None,
            "seniority_ceiling": None,
        }
    )
    assert criteria.location_prefs.cities == ["Los Angeles"]
    assert criteria.location_prefs.onsite_ok is False


def test_exclude_city_overrides_a_positive_city_match():
    prefs = LocationPrefs(cities=["New York"], exclude_cities=["New York"])
    assert matches_location(prefs, workplace_type="onsite", locations=["New York, NY"]) is False


def test_exclude_country_overrides_a_positive_remote_match():
    prefs = LocationPrefs(
        remote_ok=True, countries=["United States"], exclude_countries=["United States"]
    )
    assert (
        matches_location(prefs, workplace_type="remote", locations=["Remote, United States"])
        is False
    )


def test_exclude_city_is_noop_without_location_data():
    prefs = LocationPrefs(exclude_cities=["New York"])
    assert matches_location(prefs, workplace_type=None, locations=None) is None


def test_exclude_city_does_nothing_when_not_mentioned():
    prefs = LocationPrefs(cities=["Seattle"], exclude_cities=["New York"])
    assert matches_location(prefs, workplace_type="onsite", locations=["Seattle, WA"]) is True


def test_exclude_city_applies_regardless_of_hybrid_onsite_flags():
    # A hard "no" isn't subject to hybrid_ok/onsite_ok the way a positive
    # city match is -- there's no workplace type that makes an excluded
    # city acceptable.
    prefs = LocationPrefs(exclude_cities=["New York"], hybrid_ok=True, onsite_ok=True)
    assert matches_location(prefs, workplace_type="hybrid", locations=["New York, NY"]) is False
