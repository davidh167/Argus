"""Test setup: a dedicated, throwaway database, and no live API calls.

Tests run against `<DATABASE_NAME>_test`, which is dropped, recreated and migrated at the start of
every run, so they never read or write real data and behave the same on any machine with a Postgres
server (the one from docker-compose.yml is enough).
"""

from pathlib import Path

import psycopg
import pytest
from alembic.config import Config
from sqlalchemy.engine import make_url

from alembic import command
from app.config import settings

ROOT = Path(__file__).resolve().parents[1]

# Stage 1 title phrases now live in the profile. Tests that capture titles like "ML Engineer" put these
# in their snapshot's criteria so Stage 1 routes them as it did when the lists were built in.
TITLE_RULES = {
    "title_fast_track": [
        "research engineer",
        "ml engineer",
        "nlp engineer",
        "ai engineer",
        "machine learning",
        "knowledge graph",
        "knowledge engineer",
        "knowledge base",
        "retrieval",
        "llm",
        "language model",
    ],
    "title_skip": [
        "solutions engineer",
        "account executive",
        "account manager",
        "sales engineer",
        "developer advocate",
        "devrel",
        "site reliability engineer",
        "data center",
    ],
}


def pytest_configure(config):
    base = make_url(settings.database_url)
    test_db = f"{base.database}_test"
    if test_db == base.database:
        raise RuntimeError("refusing to run the tests against the configured database")
    admin = base.set(drivername="postgresql", database="postgres").render_as_string(
        hide_password=False
    )
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{test_db}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{test_db}"')
    # Must happen before anything imports app.db, which builds its engine from this URL.
    settings.database_url = base.set(database=test_db).render_as_string(hide_password=False)
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")


@pytest.fixture(autouse=True)
def _no_real_anthropic_calls(monkeypatch):
    """The dev .env may hold a real API key, so any test that captures a posting with an
    "analyze"-bucket title would otherwise make a live (paid) Stage 2 call. Tests that need a
    configured key set their own (patch.object(..., "fake-key")) together with a mocked client."""
    monkeypatch.setattr(settings, "anthropic_api_key", "")
