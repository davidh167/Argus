import pytest
from sqlalchemy.orm import Session

from app.db import engine
from app.schemas.profile import CreateProfileVersionRequest, ProfileSnapshotData
from app.services.profile import create_profile_version, get_current_profile, list_profile_versions

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
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def test_list_profile_versions_orders_newest_first(db):
    older = create_profile_version(db, snapshot=EMPTY_SNAPSHOT, changelog="seed")
    newer = create_profile_version(db, snapshot=EMPTY_SNAPSHOT, changelog="added a project")

    versions = list_profile_versions(db)
    ids_in_order = [v.id for v in versions if v.id in {older.id, newer.id}]
    assert ids_in_order == [newer.id, older.id]


def test_list_profile_versions_includes_changelog(db):
    version = create_profile_version(db, snapshot=EMPTY_SNAPSHOT, changelog="a specific note")
    found = next(v for v in list_profile_versions(db) if v.id == version.id)
    assert found.changelog == "a specific note"


def test_create_profile_version_request_round_trips_through_the_full_pipeline(db):
    # Exercises exactly what POST /profile does (4.4): validate the
    # incoming payload, dump it back to a dict, persist it, then confirm
    # get_current_profile re-validates the same data correctly -- no
    # router-level test infra exists in this codebase, so this covers the
    # router's actual logic at the schema/service boundary instead.
    payload = CreateProfileVersionRequest.model_validate(
        {
            "snapshot": {
                **EMPTY_SNAPSHOT,
                "skills": {"core": ["Python"], "developing": ["Kubernetes"], "coursework": []},
            },
            "changelog": "Completed roadmap item: Kubernetes -- added to skills.developing",
        }
    )
    version = create_profile_version(
        db, snapshot=payload.snapshot.model_dump(mode="json"), changelog=payload.changelog
    )

    current = get_current_profile(db)
    assert current.id == version.id
    assert current.changelog == "Completed roadmap item: Kubernetes -- added to skills.developing"
    reparsed = ProfileSnapshotData.model_validate(current.snapshot)
    assert reparsed.skills.developing == ["Kubernetes"]
    assert reparsed.skills.core == ["Python"]


def test_blank_seed_template_validates_and_sets_no_placeholder_criteria():
    # The template is meant to be copied and filled in, but a stranger who seeds it unedited must
    # not end up with literal "e.g. ..." strings stored as real triage rules.
    from pathlib import Path

    import yaml

    from app.schemas.profile import ProfileSnapshotData

    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "profile_seed.yaml").read_text())
    raw.pop("changelog", None)
    criteria = ProfileSnapshotData.model_validate(raw).criteria
    assert criteria.skip_patterns == []
    assert criteria.target_signals == []
    assert criteria.title_fast_track == []
    assert criteria.title_skip == []
    assert criteria.seniority_floor is None and criteria.seniority_ceiling is None
