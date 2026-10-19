"""Triage-verdict noise floor. Compares, on the postings with a stored v1
verdict: stored v1 vs a fresh re-run of the SAME v1 prompt (control), and
fresh v1 vs v2. If v1-vs-control disagreement is about as large as
v1-vs-v2, the verdict shifts are run-to-run noise, not the prompt.
Read-only. Run after both batches have ended:
PYTHONPATH=. poetry run python research/stage2-prompt/compare_control.py"""

import json
import sys
from pathlib import Path

from anthropic import Anthropic

from app.config import settings
from app.db import SessionLocal
from app.models.posting import Posting

HERE = Path(__file__).parent
client = Anthropic(api_key=settings.anthropic_api_key)


def fetch(file):
    batch_id = json.loads((HERE / "data" / file).read_text())["batch_id"]
    status = client.messages.batches.retrieve(batch_id).processing_status
    if status != "ended":
        sys.exit(f"{file}: batch is {status}; try again later")
    out = {}
    for item in client.messages.batches.results(batch_id):
        if item.result.type == "succeeded":
            tool = next(b for b in item.result.message.content if b.type == "tool_use")
            out[int(item.custom_id.removeprefix("posting-"))] = tool.input["result"]
    return out


v2, ctrl = fetch("batch.json"), fetch("batch_control.json")
with SessionLocal() as db:
    stored = {
        p.id: p.verdicts[0].stage2_result
        for p in db.query(Posting).filter(Posting.id.in_(list(v2)))
    }
ids = sorted(i for i in v2 if i in ctrl and stored[i] is not None)


def diff(a, b):
    return sum(a[i] != b[i] for i in ids)


rank = {"skip": 0, "analyze": 1, "fast_track": 2}


def toward_fast(a, b):
    return sum(rank[b[i]] > rank[a[i]] for i in ids if a[i] != b[i])


lines = [
    f"# Triage verdict noise floor ({len(ids)} postings with a stored v1 verdict)",
    "",
    "| comparison | verdicts differ | of which more permissive |",
    "|---|---|---|",
    f"| stored v1 vs control (same v1 prompt, re-run) | {diff(stored, ctrl)} | {toward_fast(stored, ctrl)} |",
    f"| control vs v2 | {diff(ctrl, v2)} | {toward_fast(ctrl, v2)} |",
    f"| stored v1 vs v2 | {diff(stored, v2)} | {toward_fast(stored, v2)} |",
]
(HERE / "reports" / "noise_floor.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
