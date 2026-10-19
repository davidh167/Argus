"""Scores the LLM skill judgment against your own labels for the top interest-set clusters.
Labels are personal, so they live in data/labels.json (gitignored): {"skill label": "have" | "developing" | "gap"},
keyed by the cluster label the demand view shows; see labels.example.json.

DRY RUN by default: retrieves evidence, prints one full prompt and an
estimate; no API call. `--run` makes one SYNC Sonnet call per labeled cluster
(~22 calls, cents) -- user-triggered. Reads the DB, writes nothing to it.

Run: PYTHONPATH=. poetry run python research/skill-judgment/run.py [--run]"""

import json
import sys
from pathlib import Path

import judge  # noqa: E402
from anthropic import Anthropic
from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.models.posting import Posting
from app.models.verdict import Verdict
from app.schemas.profile import ProfileSnapshotData
from app.services import profile as profile_service
from app.services import roadmap
from app.services.postings import latest_verdict_id_subquery
from app.services.stage3 import STAGE3_MODEL

HERE = Path(__file__).parent
LABELS_FILE = HERE / "data" / "labels.json"
if not LABELS_FILE.exists():
    sys.exit(f"create {LABELS_FILE} first (see labels.example.json)")
GOLD = json.loads(LABELS_FILE.read_text())

with SessionLocal() as db:
    postings = list(
        db.scalars(
            select(Posting)
            .join(Verdict, Verdict.id == latest_verdict_id_subquery())
            .where(Verdict.stage2_key_skills.is_not(None))
        )
    )
    weights = roadmap.interest_weights(db, postings)
    clusters = roadmap._cluster_demand_phrases(postings, weights)
    snap = ProfileSnapshotData.model_validate(profile_service.get_current_profile(db).snapshot)

evidence = judge.evidence_items(snap)
cases = []
for c in clusters:
    label = roadmap._cluster_medoid_dense(c["texts"], c["vectors"])
    if label in GOLD:
        shortlist = judge.retrieve(label, c["texts"], evidence)
        cases.append(
            (
                label,
                judge.build_user_message(label, c["texts"], shortlist),
                {i for i, _ in shortlist},
            )
        )

print(f"{len(cases)} labeled clusters; {len(evidence)} evidence items in profile")
chars = len(judge.SYSTEM_PROMPT) * len(cases) + sum(len(m) for _, m, _ in cases)
print(f"~{chars / 4:,.0f} input tokens total (Sonnet sync pricing applies; a few cents)\n")
print(cases[0][1])

if "--run" not in sys.argv:
    print("\ndry run -- pass --run to call the API")
    sys.exit()

client = Anthropic(api_key=settings.anthropic_api_key)
results = {}
for label, message, shown in cases:
    resp = client.messages.create(
        model=STAGE3_MODEL,
        max_tokens=300,
        system=judge.SYSTEM_PROMPT,
        messages=[{"role": "user", "content": message}],
        tools=[judge.JUDGE_TOOL],
        tool_choice={"type": "tool", "name": judge.JUDGE_TOOL["name"]},
    )
    out = next(b for b in resp.content if b.type == "tool_use").input
    out["cited_unknown_ids"] = sorted(set(out["evidence_ids"]) - shown)
    results[label] = out
(HERE / "data" / "judgments.json").write_text(json.dumps(results, indent=1))

right = [l for l in results if results[l]["status"] == GOLD[l]]
lines = [f"LLM skill judgment vs the author's labels: {len(right)}/{len(results)} correct", ""]
for status in ("have", "developing", "gap"):
    of = [l for l in results if GOLD[l] == status]
    lines.append(
        f"- labeled {status}: {sum(results[l]['status'] == status for l in of)}/{len(of)} judged correctly"
    )
bad = [l for l, r in results.items() if r["cited_unknown_ids"]]
lines += ["", f"Judgments citing evidence outside their shortlist: {len(bad)}"]
(HERE / "reports").mkdir(exist_ok=True)
(HERE / "reports" / "judgment.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
