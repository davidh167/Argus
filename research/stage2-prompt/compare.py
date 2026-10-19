"""Compares v2 key_skills (from the batch submitted by build_requests.py) with
the v1 key_skills already stored for the same postings. Read-only: fetches
batch results from Anthropic, reads the DB, writes reports/v2_vs_v1.md.
Does not apply anything to the DB.

Run: PYTHONPATH=. poetry run python research/stage2-prompt/compare.py"""

import json
import re
import sys
from pathlib import Path

from anthropic import Anthropic

from app.config import settings
from app.db import SessionLocal
from app.models.posting import Posting
from app.schemas.profile import ProfileSnapshotData
from app.services import embeddings
from app.services import profile as profile_service

sys.path.insert(0, str(Path(__file__).parents[1] / "cluster-quality"))
import methods  # noqa: E402

HERE = Path(__file__).parent
batch = json.loads((HERE / "data" / "batch.json").read_text())
client = Anthropic(api_key=settings.anthropic_api_key)
status = client.messages.batches.retrieve(batch["batch_id"]).processing_status
if status != "ended":
    sys.exit(f"batch is {status}; try again later")

new, errors = {}, 0
for item in client.messages.batches.results(batch["batch_id"]):
    if item.result.type != "succeeded":
        errors += 1
        continue
    tool = next(b for b in item.result.message.content if b.type == "tool_use")
    new[int(item.custom_id.removeprefix("posting-"))] = tool.input

with SessionLocal() as db:
    postings = {p.id: p for p in db.query(Posting).filter(Posting.id.in_(list(new)))}
    old = {
        pid: (p.verdicts[0].stage2_key_skills, p.verdicts[0].stage2_result)
        for pid, p in postings.items()
    }
    snap = ProfileSnapshotData.model_validate(profile_service.get_current_profile(db).snapshot)
    titles = {pid: f"{p.title} @ {p.company.name}" for pid, p in postings.items()}

profile_vocab = {
    s.casefold()
    for s in [
        *snap.skills.core,
        *snap.skills.developing,
        *(t for p in snap.projects for t in p.stack),
    ]
}


def norm(p):
    return " ".join(p.casefold().replace("-", " ").split())


def stats(lists, cap):
    flat = [s for l in lists for s in l]
    distinct = sorted(set(flat))
    groups = {}
    for p in distinct:
        groups.setdefault(norm(p), set()).add(p)
    v = embeddings.embed_texts(distinct)
    clusters = len(set(methods.agglomerative(distinct, v, 0.80)))
    return {
        "skills per posting": f"{len(flat) / len(lists):.1f}",
        "postings at their cap (8 v1 / 12 v2)": sum(len(l) >= cap for l in lists),
        "distinct phrases": len(distinct),
        "mentions per distinct phrase": f"{len(flat) / len(distinct):.2f}",
        "compound phrases (/ ( and &)": f"{sum(bool(re.search(r'[/(&]|\\band\\b', s)) for s in distinct)} ({sum(bool(re.search(r'[/(&]|\\band\\b', s)) for s in distinct) / len(distinct):.0%})",
        "case/hyphen variant groups": sum(len(g) > 1 for g in groups.values()),
        "clusters @0.80 (avg-link)": clusters,
        "exact matches to profile vocabulary": sum(
            norm(s) in {norm(x) for x in profile_vocab} for s in flat
        ),
    }


ids = sorted(new)
s_old = stats([old[i][0] for i in ids], 8)
s_new = stats([new[i]["key_skills"] for i in ids], 12)
comparable = [
    i for i in ids if old[i][1] is not None
]  # manual captures never got a stored v1 triage
disagree = [(i, old[i][1], new[i]["result"]) for i in comparable if old[i][1] != new[i]["result"]]

out = [
    f"# Stage 2 key_skills: v1 vs v2 on {len(ids)} interest-set postings ({errors} batch errors)",
    "",
    "| metric | v1 (stored) | v2 (candidate) |",
    "|---|---|---|",
]
for k in s_old:
    out.append(f"| {k} | {s_old[k]} | {s_new[k]} |")
out += [
    "",
    f"Triage verdict changed on {len(disagree)} of {len(comparable)} postings that had a stored v1 verdict.",
]
(HERE / "reports").mkdir(exist_ok=True)
(HERE / "reports" / "v2_vs_v1.md").write_text("\n".join(out) + "\n")
print("\n".join(out[:20]))
