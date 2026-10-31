"""Is another model's Stage 2 triage as consistent as the production model's? Compares verdicts on the
postings in the interest set between: the stored production verdict ("stored"), a re-run of the same
prompt on the production model (batch_control.json, "control"), and one or more runs on another model
saved by `build_requests.py --model ID --tag NAME`. The yardstick is the production model's own
run-to-run disagreement (stored vs control): a candidate whose runs agree with each other and with the
production runs about as well as those agree with each other is not worse. Also reports errored
requests, refusals, truncated outputs and token use. Read-only; nothing is written to the database.
Run (after the batches have ended): PYTHONPATH=. poetry run python research/stage2-prompt/compare_models.py TAG [TAG ...]"""

import collections
import itertools
import json
import sys
from pathlib import Path

from anthropic import Anthropic

from app.config import settings
from app.db import SessionLocal
from app.models.posting import Posting

HERE = Path(__file__).parent


def disagreement(a: dict[int, str], b: dict[int, str]) -> tuple[int, int]:
    """(postings whose verdicts differ, postings both runs have a verdict for)."""
    common = sorted(set(a) & set(b))
    return sum(a[i] != b[i] for i in common), len(common)


def transitions(a: dict[int, str], b: dict[int, str]) -> list[str]:
    """One line per (verdict in a -> verdict in b) pair, most common first, for postings both runs cover."""
    common = sorted(set(a) & set(b))
    counts = collections.Counter((a[i], b[i]) for i in common)
    return [f"    {x} -> {y}: {n}" for (x, y), n in counts.most_common()]


def summarize(batch_file: str, client: Anthropic):
    batch_id = json.loads((HERE / "data" / batch_file).read_text())["batch_id"]
    status = client.messages.batches.retrieve(batch_id).processing_status
    if status != "ended":
        sys.exit(f"{batch_file}: batch is {status}; try again later")
    verdicts, problems, usage = {}, [], {"in": 0, "out": 0, "truncated": 0, "refusals": 0}
    for item in client.messages.batches.results(batch_id):
        pid = int(item.custom_id.removeprefix("posting-"))
        if item.result.type != "succeeded":
            problems.append((pid, item.result.type, str(getattr(item.result, "error", ""))[:200]))
            continue
        message = item.result.message
        usage["in"] += message.usage.input_tokens
        usage["out"] += message.usage.output_tokens
        if message.stop_reason == "refusal":
            usage["refusals"] += 1
            continue
        if message.stop_reason == "max_tokens":
            usage["truncated"] += 1
        tool = next((b for b in message.content if b.type == "tool_use"), None)
        if tool is None:
            problems.append((pid, "no_tool_call", message.stop_reason))
            continue
        verdicts[pid] = tool.input["result"]
    return verdicts, problems, usage


if __name__ == "__main__":
    tags = sys.argv[1:]
    if not tags:
        sys.exit("usage: compare_models.py TAG [TAG ...]")
    client = Anthropic(api_key=settings.anthropic_api_key)
    runs, lines = {}, []
    runs["control"], control_problems, _ = summarize("batch_control.json", client)
    for tag in tags:
        runs[tag], problems, usage = summarize(f"batch_{tag}.json", client)
        lines.append(
            f"- {tag}: {len(runs[tag])} verdicts, {len(problems)} problems, "
            f"{usage['refusals']} refusals, {usage['truncated']} truncated at max_tokens, "
            f"{usage['in']:,} input / {usage['out']:,} output tokens"
        )
        lines += [f"    problem: {p}" for p in problems[:3]]
    with SessionLocal() as db:
        runs["stored"] = {
            p.id: p.verdicts[0].stage2_result
            for p in db.query(Posting).filter(Posting.id.in_(list(runs["control"])))
            if p.verdicts[0].stage2_result is not None
        }
    table = ["", "| comparison | verdicts differ | of |", "|---|---|---|"]
    for a, b in itertools.combinations(["stored", "control", *tags], 2):
        n, of = disagreement(runs[a], runs[b])
        label = f"{a} vs {b}"
        if (a, b) == ("stored", "control"):
            label += " (production model's own noise)"
        table.append(f"| {label} | {n} | {of} |")
    moves = []
    for a, b in itertools.combinations(["stored", *tags], 2):
        moves += [
            f"{a} -> {b} (rows: verdict in {a}, then in {b}):",
            *transitions(runs[a], runs[b]),
        ]
    print(
        "\n".join(
            ["# Stage 2 model comparison", "", *lines, *table, "", "## Verdict transitions", *moves]
        )
    )
