import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import engine
from app.models.posting import Posting
from app.models.verdict import Verdict
from app.schemas.stage3 import Stage3Result
from app.services.roadmap import readiness_view
from scripts import seed_demo

EXPECTED_COUNTS = {
    "profile_versions": 1,
    "companies": 8,
    "postings": 21,
    "verdicts": 21,
    "stage2_results": 5,
    "stage2_key_skills": 16,
    "stage3_results": 12,
    "starred": 4,
    "applications": 3,
    "roadmap_items": 1,
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


def test_seed_counts_and_idempotent(db):
    first = seed_demo.seed(db)
    assert first == EXPECTED_COUNTS
    assert seed_demo.seed(db) == EXPECTED_COUNTS


def test_stage3_results_validate_against_schema(db):
    seed_demo.seed(db)
    results = [
        v.stage3_result
        for v in db.scalars(select(Verdict).where(Verdict.stage3_result.is_not(None)))
    ]
    assert len(results) == EXPECTED_COUNTS["stage3_results"]
    parsed = [Stage3Result.model_validate(r) for r in results]
    assert {r.verdict for r in parsed} == {"Strong Fit", "Moderate Fit", "Weak Fit", "Skip"}
    assert {"No", "Borderline", "Yes"} <= {r.seniority_match for r in parsed}
    assert any(r.hard_gaps for r in parsed)


def test_stage1_results_cover_every_route(db):
    seed_demo.seed(db)
    routes = set(db.scalars(select(Verdict.stage1_result)))
    assert routes == {"skip", "fast_track", "analyze"}


def test_readiness_view_has_every_tier(db):
    seed_demo.seed(db)
    view = readiness_view(db)
    assert view.apply_now
    assert view.solid_match
    assert view.stretch
    assert view.build_toward
    # The staff-level role is past the candidate's range, so it is in no tier.
    tiered = {e.title for tier in (view.apply_now, view.solid_match, view.stretch) for e in tier}
    assert "Staff Data Platform Engineer" not in tiered


def test_postings_are_fictional_and_dated_relative_to_today(db):
    seed_demo.seed(db)
    postings = list(db.scalars(select(Posting)))
    assert all(".example.com/" in p.url for p in postings)
    assert any(p.comp_min for p in postings)
    assert all(p.company.crawl_enabled is False for p in postings)


def test_demo_database_url_replaces_only_the_database():
    url = seed_demo.demo_database_url("postgresql+psycopg://u:pw@localhost:5432/job_search")
    assert url.database == "argus_demo"
    assert url.host == "localhost" and url.username == "u"


def test_guard_accepts_demo_database():
    seed_demo.check_target_database("argus_demo", "job_search")


@pytest.mark.parametrize(
    "target, main",
    [
        ("job_search", "job_search"),
        ("argus", "job_search"),
        ("argus_demo_backup", "job_search"),
        ("job_search_demo", "job_search_demo"),
        (None, "job_search"),
    ],
)
def test_guard_refuses_other_databases(target, main):
    with pytest.raises(SystemExit):
        seed_demo.check_target_database(target, main)
