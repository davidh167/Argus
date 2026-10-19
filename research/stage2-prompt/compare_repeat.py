"""Do two identical Stage 2 batches agree with each other? Reads two batches
saved by build_requests.py (data/batch_<tag>.json), compares triage verdicts,
and reports any errored requests (e.g. the API rejecting temperature).
Read-only.
Run: PYTHONPATH=. poetry run python research/stage2-prompt/compare_repeat.py TAG_A TAG_B"""

import json
import sys
from pathlib import Path

from anthropic import Anthropic

from app.config import settings

HERE = Path(__file__).parent
client = Anthropic(api_key=settings.anthropic_api_key)


def fetch(tag):
    batch_id = json.loads((HERE / "data" / f"batch_{tag}.json").read_text())["batch_id"]
    status = client.messages.batches.retrieve(batch_id).processing_status
    if status != "ended":
        sys.exit(f"{tag}: batch is {status}; try again later")
    ok, errors = {}, []
    for item in client.messages.batches.results(batch_id):
        pid = int(item.custom_id.removeprefix("posting-"))
        if item.result.type == "succeeded":
            ok[pid] = next(b for b in item.result.message.content if b.type == "tool_use").input[
                "result"
            ]
        else:
            errors.append((pid, item.result.type, str(getattr(item.result, "error", ""))[:300]))
    return ok, errors


(a, ea), (b, eb) = fetch(sys.argv[1]), fetch(sys.argv[2])
both = sorted(set(a) & set(b))
diff = [p for p in both if a[p] != b[p]]
print(
    f"{sys.argv[1]}: {len(a)} ok, {len(ea)} errored | {sys.argv[2]}: {len(b)} ok, {len(eb)} errored"
)
for e in (ea + eb)[:3]:
    print("  error:", e)
print(f"verdicts differing between the two runs: {len(diff)} of {len(both)}")
