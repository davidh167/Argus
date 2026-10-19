"""Step 1: freeze a labeled snapshot of the corpus (read-only on the DB).

Writes data/snapshot.jsonl (one posting per line) + data/meta.json. Everything
after this step reads the snapshot, never the live DB, so a report is
reproducible from the exact data it was computed on.

Labels (see README "Labels"): a posting is a POSITIVE when it carries an
explicit interest signal -- manual capture, star, or an active application --
via app.services.roadmap.interest_weights. Everything else is UNLABELED (not
negative). Dismissed postings are dropped: they are the only explicit
negatives and there are currently none.

Run from the project root:  PYTHONPATH=. poetry run python research/ranking-eval/export_snapshot.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal
from app.models.posting import Posting
from app.models.verdict import Verdict
from app.schemas.profile import ProfileSnapshotData
from app.services import profile as profile_service
from app.services import relevance, roadmap
from app.services.postings import latest_verdict_id_subquery

DATA = Path(__file__).parent / "data"


def main() -> None:
    db = SessionLocal()
    rows = db.execute(
        select(Posting, Verdict)
        .join(Verdict, Verdict.id == latest_verdict_id_subquery())
        .where(Posting.dismissed_at.is_(None), Posting.normalized_jd_text.is_not(None))
    ).all()
    postings = [p for p, _ in rows]
    weights = roadmap.interest_weights(db, postings)

    profile_version = profile_service.get_current_profile(db)
    profile = ProfileSnapshotData.model_validate(profile_version.snapshot)

    DATA.mkdir(exist_ok=True)
    n_pos = 0
    with open(DATA / "snapshot.jsonl", "w") as f:
        for p, v in rows:
            positive = weights[p.id] >= roadmap.INTEREST_SET_MIN_WEIGHT
            n_pos += positive
            f.write(
                json.dumps(
                    {
                        "id": p.id,
                        "title": p.title,
                        "company": p.company.name,
                        "source": p.source,
                        "closed": p.closed_at is not None,
                        "text": p.normalized_jd_text,
                        "positive": positive,
                        "stage1": v.stage1_result,
                        "stage2": v.stage2_result,
                        "relevance_score": p.relevance_score,
                    }
                )
                + "\n"
            )
    meta = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "n_postings": len(rows),
        "n_positive": n_pos,
        "profile_version_id": profile_version.id,
        "profile_query": relevance.profile_query_text(profile),
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "profile_query"}, indent=2))
    db.close()


if __name__ == "__main__":
    main()
