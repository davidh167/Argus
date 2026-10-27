"""Builds Stage 2 Batch API requests for the interest-set postings using the
v2 key_skills prompt. DRY RUN by default: prints the request count and a token/cost
estimate and writes nothing to the API or DB. Submitting is a separate, explicit
`--submit` that the user runs (it costs money); the batch id is saved to
data/batch.json. Results are NOT applied to the DB -- compare.py reads them.

Run: PYTHONPATH=. poetry run python research/stage2-prompt/build_requests.py [--control | --temp0 | --model ID --tag NAME] [--tag NAME] [--submit]

--model ID re-runs the PRODUCTION prompt on another model (e.g. claude-haiku-5-5), which needs --tag so
it cannot overwrite another batch's id; --max-tokens N overrides the production output limit (256) for
that run. compare_models.py reads the results."""

import json
import sys
from pathlib import Path

import candidate
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.models.posting import Posting
from app.models.verdict import Verdict
from app.services import roadmap
from app.services.postings import latest_verdict_id_subquery
from app.services.stage2_batch import build_request

HERE = Path(__file__).parent
CONTROL = "--control" in sys.argv  # re-run the PRODUCTION prompt
TEMP0 = "--temp0" in sys.argv  # production prompt, with temperature=0 sent as a raw request field
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else None
MODEL = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else None
if MODEL and not TAG:
    sys.exit("--model needs --tag NAME (it names the saved batch id)")
MAX_TOKENS = int(sys.argv[sys.argv.index("--max-tokens") + 1]) if "--max-tokens" in sys.argv else None
# Input $/MTok at the Batch API rate (half the standard rate), and how many tokens the model's
# tokenizer produces relative to the chars/4 estimate below (Haiku 5.5's is ~30% larger).
BATCH_INPUT_RATE = {"claude-haiku-5-5": 0.05}.get(MODEL, 0.50)
TOKEN_FACTOR = 1.3 if MODEL == "claude-haiku-5-5" else 1.0
BATCH_FILE = (
    f"batch_{TAG}.json" if TAG else ("batch_control.json" if CONTROL or TEMP0 else "batch.json")
)

with SessionLocal() as db:
    postings = list(
        db.scalars(
            select(Posting)
            .join(Verdict, Verdict.id == latest_verdict_id_subquery())
            .where(Verdict.stage2_key_skills.is_not(None))
        )
    )
    weights = roadmap.interest_weights(db, postings)
    picked = [p for p in postings if weights[p.id] >= roadmap.INTEREST_SET_MIN_WEIGHT]
    requests = []
    for p in picked:
        base = build_request(db, p)
        if MODEL:  # production prompt and tool, different model: nothing else changes
            params = dict(base["params"])
            params["model"] = MODEL
            if MAX_TOKENS:
                params["max_tokens"] = MAX_TOKENS
            requests.append(Request(custom_id=base["custom_id"], params=params))
            continue
        if TEMP0:
            # anthropic 1.6's typed create() has no temperature argument, so this goes in the
            # batch request's raw params. Whether the API accepts it for Haiku 4.5 is exactly
            # what this run checks (errored results == rejected).
            params = dict(base["params"])
            params["temperature"] = 0
            requests.append(Request(custom_id=base["custom_id"], params=params))
            continue
        if CONTROL:  # unchanged production prompt: measures run-to-run noise
            requests.append(base)
            continue
        params = dict(base["params"])
        params["system"] = candidate.SYSTEM_PROMPT_V2
        params["tools"] = [candidate.STAGE2_TOOL_V2]
        requests.append(
            Request(custom_id=base["custom_id"], params=MessageCreateParamsNonStreaming(**params))
        )

chars = sum(
    len(r["params"]["system"]) + len(r["params"]["messages"][0]["content"]) for r in requests
)
est_tokens = chars / 4 * TOKEN_FACTOR
print(
    f"{len(requests)} requests; ~{est_tokens:,.0f} input tokens; "
    f"~${est_tokens / 1e6 * BATCH_INPUT_RATE:.3f} input at the batch rate for {MODEL or 'the production model'} "
    "(estimate, output extra but small)"
)

if "--submit" in sys.argv:
    batch = Anthropic(api_key=settings.anthropic_api_key).messages.batches.create(requests=requests)
    (HERE / "data" / BATCH_FILE).write_text(json.dumps({"batch_id": batch.id, "n": len(requests)}))
    print("submitted", batch.id)
else:
    print("dry run -- pass --submit to send (costs money)")
