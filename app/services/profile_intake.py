"""Profile intake chat: a conversational alternative to the desktop Profile
tab's structured "Edit profile" form. The user describes their background
in free text; the model -- which is given ProfileSnapshotData's shape and
the CURRENT snapshot -- reconciles it against what's already there,
surfacing overlaps/conflicts/ambiguous placement as plain-text questions
before proposing a concrete update.

Unlike every other Anthropic call in this codebase (Stage 2, Stage 3,
generate_roadmap_brief), this one does NOT force a tool call -- the model
should converse freely (tool_choice="auto") and only call
PROFILE_INTAKE_RESULT_TOOL once it's ready to propose a full replacement
snapshot.

chat() itself is a pure computation (given a message history + new
message, what's the model's next turn) and does not touch the database --
easy to unit-test with a mocked Anthropic client. The conversation log
itself IS persisted (ProfileIntakeMessage, one append-only log -- a
single ongoing conversation, not multiple sessions, since this is a
single-user app with one profile to reconcile against): run_chat_turn
below loads it, calls chat(), and appends both sides of the turn. Only a
user-approved PROPOSAL ever gets written to the actual profile, through
the existing profile_service.create_profile_version path (unchanged by
this file) -- persisting the conversation log is separate from that and
does not itself change the profile.

Deliberately field-agnostic: nothing here assumes what field the user works in.
(Stage 1 title rules live in the profile too: criteria.title_fast_track / title_skip.)
"""

from typing import Literal

from anthropic import Anthropic
from anthropic.types import MessageParam, ToolParam
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.profile_intake_message import ProfileIntakeMessage
from app.schemas.profile import ChatMessage, ProfileSnapshotData
from app.services.stage2 import Stage2UnavailableError

PROFILE_INTAKE_MODEL = "claude-sonnet-5"

_WEIGHT_ENUM = {"type": "string", "enum": ["high", "medium", "low"]}

PROFILE_INTAKE_RESULT_TOOL: ToolParam = {
    "name": "propose_profile_update",
    "description": (
        "Propose a full replacement profile snapshot once you have enough information "
        "and every ambiguity has been resolved through conversation. Never call this "
        "while you still have an open question -- ask it as a normal message instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "snapshot": {
                "type": "object",
                "properties": {
                    "experience": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "company": {"type": "string"},
                                "title": {"type": "string"},
                                "date_range": {"type": "string"},
                                "signals": {"type": "array", "items": {"type": "string"}},
                                "weight": _WEIGHT_ENUM,
                            },
                            "required": ["company", "title", "date_range", "signals", "weight"],
                            "additionalProperties": False,
                        },
                    },
                    "projects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "stack": {"type": "array", "items": {"type": "string"}},
                                "key_metrics": {"type": "array", "items": {"type": "string"}},
                                "visibility": {
                                    "type": "string",
                                    "enum": ["private", "public", "deployed"],
                                },
                                "weight": _WEIGHT_ENUM,
                            },
                            "required": [
                                "name",
                                "description",
                                "stack",
                                "key_metrics",
                                "visibility",
                                "weight",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "skills": {
                        "type": "object",
                        "properties": {
                            "core": {"type": "array", "items": {"type": "string"}},
                            "developing": {"type": "array", "items": {"type": "string"}},
                            "coursework": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["core", "developing", "coursework"],
                        "additionalProperties": False,
                    },
                    "credentials": {
                        "type": "object",
                        "properties": {
                            "education": {"type": "array", "items": {"type": "string"}},
                            "oss_contributions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "repo": {"type": "string"},
                                        "type": {
                                            "type": "string",
                                            "enum": ["docs", "bug_fix", "feature"],
                                        },
                                        "merged": {"type": "boolean"},
                                        "description": {"type": "string"},
                                        "relevance": {"type": "string"},
                                    },
                                    "required": [
                                        "repo",
                                        "type",
                                        "merged",
                                        "description",
                                        "relevance",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                            "publications": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["education", "oss_contributions", "publications"],
                        "additionalProperties": False,
                    },
                    "criteria": {
                        "type": "object",
                        "properties": {
                            "skip_patterns": {"type": "array", "items": {"type": "string"}},
                            "target_signals": {"type": "array", "items": {"type": "string"}},
                            "title_fast_track": {"type": "array", "items": {"type": "string"}},
                            "title_skip": {"type": "array", "items": {"type": "string"}},
                            "location_prefs": {
                                "type": "object",
                                "properties": {
                                    "cities": {"type": "array", "items": {"type": "string"}},
                                    "countries": {"type": "array", "items": {"type": "string"}},
                                    "exclude_cities": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "exclude_countries": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "remote_ok": {"type": "boolean"},
                                    "hybrid_ok": {"type": "boolean"},
                                    "onsite_ok": {"type": "boolean"},
                                },
                                "required": [
                                    "cities",
                                    "countries",
                                    "exclude_cities",
                                    "exclude_countries",
                                    "remote_ok",
                                    "hybrid_ok",
                                    "onsite_ok",
                                ],
                                "additionalProperties": False,
                            },
                            "seniority_floor": {"type": ["string", "null"]},
                            "seniority_ceiling": {"type": ["string", "null"]},
                        },
                        "required": [
                            "skip_patterns",
                            "target_signals",
                            "title_fast_track",
                            "title_skip",
                            "location_prefs",
                            "seniority_floor",
                            "seniority_ceiling",
                        ],
                        "additionalProperties": False,
                    },
                },
                "required": ["experience", "projects", "skills", "credentials", "criteria"],
                "additionalProperties": False,
            },
            "changelog": {
                "type": "string",
                "description": "What changed and why, in the same style as any other profile version's changelog.",
            },
            "open_questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Caveats worth flagging alongside this proposal without blocking on them. Empty if none.",
            },
        },
        "required": ["snapshot", "changelog", "open_questions"],
        "additionalProperties": False,
    },
    # Not strict, unlike every other tool schema in this codebase: this
    # schema's nesting (experience/
    # projects/skills/credentials.oss_contributions/criteria.location_prefs)
    # is too large for Anthropic's strict-mode grammar compiler ("The
    # compiled grammar is too large" 400 error). Pydantic validation on
    # the response (ProfileIntakeChatResponse.proposed_snapshot:
    # ProfileSnapshotData) is the safety net instead -- a non-conforming
    # response fails loudly there rather than silently succeeding.
}

SYSTEM_PROMPT = """You help a job seeker build and maintain their structured profile \
through conversation. The profile has five sections: experience (roles, each \
weighted high/medium/low by how strongly it should count), projects (same \
weighting, plus a visibility level), skills (three tiers by demonstrated \
strength -- core: proven in production or substantial work; developing: in \
progress, has supporting evidence; coursework: formally learned, not yet \
demonstrated in practice), credentials (education, open-source contributions, \
publications), and criteria (role-matching preferences: skip patterns, target \
signals, location preferences, seniority range).

This profile can belong to someone in any field -- do not assume a particular \
industry, role type, or skill set. Match your questions and reasoning to \
whatever the user actually describes.

You will be shown the user's CURRENT profile snapshot. When they describe new \
background information:
- Compare it against the current snapshot explicitly. Point out overlaps (the \
same experience or skill already listed, possibly differently), inconsistencies \
(conflicting dates or details), and ambiguous placement (which skill tier, \
which section) -- in plain conversation, before proposing anything.
- Ask a direct question whenever placement is genuinely ambiguous. Do not \
guess and do not call the proposal tool while a real ambiguity is still open.
- Never fabricate experience, projects, or skills beyond what the user actually \
described.
- Once things are resolved enough, call propose_profile_update with the full \
replacement snapshot (every section, not just what changed), a changelog \
explaining what changed and why, and any remaining open_questions worth \
flagging without blocking on them."""


def _snapshot_summary_block(snapshot: ProfileSnapshotData) -> str:
    return (
        "Current profile snapshot (trusted, already saved -- reconcile new "
        f"information against this):\n{snapshot.model_dump_json(indent=2)}"
    )


def chat(
    current_snapshot: ProfileSnapshotData,
    message_history: list[ChatMessage],
    user_message: str,
) -> dict:
    """Returns {assistant_message, proposed_snapshot, changelog, open_questions} --
    exactly one of assistant_message/proposed_snapshot is set. Purely computes
    the turn's result; never touches the database (see module docstring for
    why -- persistence only ever happens through the existing
    create_profile_version path, after the user reviews and saves)."""
    if not settings.anthropic_api_key:
        raise Stage2UnavailableError("ANTHROPIC_API_KEY not configured")
    client = Anthropic(api_key=settings.anthropic_api_key)

    messages: list[MessageParam] = [{"role": m.role, "content": m.content} for m in message_history]
    messages.append(
        {
            "role": "user",
            "content": f"{_snapshot_summary_block(current_snapshot)}\n\n{user_message}",
        }
    )

    response = client.messages.create(
        model=PROFILE_INTAKE_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=messages,
        tools=[PROFILE_INTAKE_RESULT_TOOL],
        tool_choice={"type": "auto"},
    )

    tool_use = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use is not None:
        return {
            "assistant_message": None,
            "proposed_snapshot": tool_use.input["snapshot"],
            "changelog": tool_use.input["changelog"],
            "open_questions": tool_use.input["open_questions"],
        }

    text_block = next(b for b in response.content if b.type == "text")
    return {
        "assistant_message": text_block.text,
        "proposed_snapshot": None,
        "changelog": None,
        "open_questions": [],
    }


def list_messages(db: Session) -> list[ProfileIntakeMessage]:
    """The persisted conversation, oldest first -- what the desktop chat
    panel renders when it opens, so a prior conversation resumes rather
    than starting blank."""
    return list(db.scalars(select(ProfileIntakeMessage).order_by(ProfileIntakeMessage.id)))


def append_message(
    db: Session, role: Literal["user", "assistant"], content: str
) -> ProfileIntakeMessage:
    message = ProfileIntakeMessage(role=role, content=content)
    db.add(message)
    db.flush()
    return message


def clear_messages(db: Session) -> None:
    """Starts a fresh conversation on request -- e.g. the desktop "Start
    over" button. Does not touch the profile itself, only this log."""
    db.query(ProfileIntakeMessage).delete()


def run_chat_turn(db: Session, current_snapshot: ProfileSnapshotData, user_message: str) -> dict:
    """Loads the persisted conversation, computes the next turn via chat()
    above, and persists both sides of it -- the actual entry point for
    POST /profile/intake-chat. A proposal turn has no plain-text assistant
    reply from the API, so its persisted form is a short synthesized
    summary (same convention the desktop client would otherwise have had
    to invent itself), enough for the model to have real continuity in
    later turns without needing to store raw tool-call content blocks."""
    history = [ChatMessage(role=m.role, content=m.content) for m in list_messages(db)]
    result = chat(current_snapshot, history, user_message)

    append_message(db, "user", user_message)
    if result["proposed_snapshot"] is not None:
        append_message(db, "assistant", f"Proposed update: {result['changelog']}")
    else:
        append_message(db, "assistant", result["assistant_message"])

    return result
