"""Read-only export of stage2 key_skills phrases per posting, with each
posting's interest weight, to data/phrases.json. No writes to the DB."""

import json
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal
from app.models.posting import Posting
from app.models.verdict import Verdict
from app.services import roadmap
from app.services.postings import latest_verdict_id_subquery

OUT = Path(__file__).parent / "data" / "phrases.json"

with SessionLocal() as db:
    postings = list(
        db.scalars(
            select(Posting)
            .join(Verdict, Verdict.id == latest_verdict_id_subquery())
            .where(Verdict.stage2_key_skills.is_not(None))
        )
    )
    weights = roadmap.interest_weights(db, postings)
    rows = [
        {"id": p.id, "interest": weights[p.id], "skills": list(p.verdicts[0].stage2_key_skills)}
        for p in postings
    ]
OUT.write_text(json.dumps(rows))
interest = [r for r in rows if r["interest"] >= roadmap.INTEREST_SET_MIN_WEIGHT]
print(
    len(rows),
    "postings;",
    len(interest),
    "interest;",
    len({s for r in interest for s in r["skills"]}),
    "distinct interest phrases;",
    len({s for r in rows for s in r["skills"]}),
    "distinct overall",
)
