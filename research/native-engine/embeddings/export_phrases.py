"""Exports the distinct stage2_key_skills phrases market_demand embeds (read-only on the DB).

Writes one phrase per line to research/native-engine/data/embedding_phrases.txt (gitignored: the
phrases come from personal postings), using the same selection and de-duplication market_demand
applies: the latest verdict of every posting, one text per normalized phrase.

Run it yourself (it reads the real DB):
  PYTHONPATH=. poetry run python research/native-engine/embeddings/export_phrases.py
"""

from pathlib import Path
from typing import cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models.posting import Posting
from app.models.verdict import Verdict
from app.services.postings import latest_verdict_id_subquery
from app.services.roadmap import _normalize_phrase

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "embedding_phrases.txt"


def distinct_phrases(db: Session) -> list[str]:
    rows = db.execute(
        select(Verdict.stage2_key_skills)
        .select_from(Posting)
        .join(Verdict, Verdict.id == latest_verdict_id_subquery())
        .where(Verdict.stage2_key_skills.is_not(None))
    ).scalars()
    first_text: dict[str, str] = {}
    for skills in rows:
        for skill in cast(list, skills):
            first_text.setdefault(_normalize_phrase(skill), skill)
    return sorted(first_text.values())


def main() -> None:
    with SessionLocal() as db:
        phrases = distinct_phrases(db)
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUT.write_text("\n".join(phrases) + "\n")
    print(f"{len(phrases)} distinct phrases -> {DEFAULT_OUT}")


if __name__ == "__main__":
    main()
