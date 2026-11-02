"""The throwaway Postgres database the benchmark runs against, and loading a corpus into it.

The harness never touches the configured database: it creates `<configured name>_bench` (or an
explicit name), refuses any name equal to the configured one, points app.config at it BEFORE app.db
builds its engine, migrates it, and drops it when done.
"""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
from sqlalchemy import insert, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from bench_corpus import Corpus

ROOT = Path(__file__).resolve().parents[2]
INSERT_CHUNK = 1000


def bench_database_name(configured: str, requested: str | None = None) -> str:
    name = requested or f"{configured}_bench"
    if name == configured:
        raise ValueError(f"refusing to benchmark against the configured database {configured!r}")
    return name


@contextmanager
def throwaway_database(requested: str | None = None) -> Iterator[str]:
    """Create, migrate and (on exit) drop the benchmark database; yields its name. While inside,
    app.config.settings.database_url points at it, so everything under app/ uses it."""
    if "app.db" in sys.modules:
        raise RuntimeError("app.db is already imported: its engine is bound to the configured URL")
    from alembic.config import Config

    from alembic import command
    from app.config import settings

    base = make_url(settings.database_url)
    name = bench_database_name(str(base.database), requested)
    admin = base.set(drivername="postgresql", database="postgres").render_as_string(
        hide_password=False
    )
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{name}"')
    try:
        settings.database_url = base.set(database=name).render_as_string(hide_password=False)
        command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
        yield name
    finally:
        if "app.db" in sys.modules:
            sys.modules["app.db"].engine.dispose()
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def load_corpus(db: Session, corpus: Corpus) -> None:
    """Insert the corpus's companies, postings, one verdict per posting (carrying the key skills),
    and a profile version. Flushes; the caller commits. raw_jd_text is set equal to the normalized
    text (real raw text is longer, so ORM loads of Posting are a lower bound)."""
    from app.models.company import Company
    from app.models.posting import Posting
    from app.models.verdict import Verdict
    from app.schemas.profile import (
        Criteria,
        ExperienceEntry,
        ProfileSnapshotData,
        ProjectEntry,
        Skills,
    )
    from app.services.profile import create_profile_version

    snapshot = ProfileSnapshotData(
        experience=[
            ExperienceEntry(
                company="Bench Co", title="Engineer", date_range="2020 -- 2024", signals=[]
            )
        ],
        projects=[
            ProjectEntry(
                name="Bench project", description="Benchmark profile.", stack=corpus.project_stack
            )
        ],
        skills=Skills(core=corpus.skills_core, developing=corpus.skills_developing),
        criteria=Criteria(target_signals=corpus.target_signals),
    )
    profile = create_profile_version(db, snapshot.model_dump(mode="json"), "benchmark profile")

    names = sorted({p.company for p in corpus.postings})
    company_ids: dict[str, int] = {}
    for start in range(0, len(names), INSERT_CHUNK):
        rows = db.execute(
            insert(Company).returning(Company.id, Company.name),
            [{"name": n} for n in names[start : start + INSERT_CHUNK]],
        )
        company_ids.update({name: cid for cid, name in rows})

    for start in range(0, len(corpus.postings), INSERT_CHUNK):
        chunk = corpus.postings[start : start + INSERT_CHUNK]
        db.execute(
            insert(Posting),
            [
                {
                    "id": p.id,
                    "company_id": company_ids[p.company],
                    "title": p.title,
                    "raw_jd_text": p.text,
                    "normalized_jd_text": p.text,
                    "source": p.source,
                    "starred": p.positive and p.source != "manual_capture",
                }
                for p in chunk
            ],
        )
        db.execute(
            insert(Verdict),
            [
                {
                    "posting_id": p.id,
                    "profile_version_id": profile.id,
                    "stage1_result": "analyze",
                    "stage2_result": "analyze",
                    "stage2_key_skills": p.key_skills,
                }
                for p in chunk
            ],
        )
    db.flush()


def clear_corpus(db: Session) -> None:
    """Remove everything load_corpus wrote, so the next corpus starts from an empty database."""
    db.execute(
        text("TRUNCATE postings, companies, verdicts, profile_versions RESTART IDENTITY CASCADE")
    )
    db.commit()
