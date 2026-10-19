from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

Weight = Literal["high", "medium", "low"]


class ExperienceEntry(BaseModel):
    company: str
    title: str
    date_range: str
    signals: list[str] = []
    weight: Weight = "medium"


class ProjectEntry(BaseModel):
    name: str
    description: str
    stack: list[str] = []
    key_metrics: list[str] = []
    visibility: Literal["private", "public", "deployed"] = "private"
    weight: Weight = "medium"


class Skills(BaseModel):
    core: list[str] = []
    developing: list[str] = []
    coursework: list[str] = []


class OssContribution(BaseModel):
    repo: str
    type: Literal["docs", "bug_fix", "feature"]
    merged: bool
    description: str
    relevance: str


class Credentials(BaseModel):
    education: list[str] = []
    oss_contributions: list[OssContribution] = []
    publications: list[str] = []


class LocationPrefs(BaseModel):
    """cities/countries are plain substrings matched case-insensitively
    against a posting's locations (e.g. "San Francisco" matches "San
    Francisco, CA") -- no geocoding or radius search. All fields default
    to permissive: the *_ok flags default True (never actively rejects a
    hybrid/onsite match on account of workplace type), and empty
    cities/countries lists impose no constraint rather than matching
    everything -- see utils.location.matches_location.

    exclude_cities/exclude_countries (5.3 follow-up) are a hard "no" --
    an exclusion match always wins over any positive match, unconditionally
    (not subject to hybrid_ok/onsite_ok the way a positive city match is,
    since there's no workplace type that makes an excluded city
    acceptable). Symmetric with cities/countries rather than a single
    generic exclude list, so "exclude this city" and "exclude this
    country" stay as unambiguous as their positive counterparts.
    """

    cities: list[str] = []
    countries: list[str] = []
    exclude_cities: list[str] = []
    exclude_countries: list[str] = []
    remote_ok: bool = True
    hybrid_ok: bool = True
    onsite_ok: bool = True


class Criteria(BaseModel):
    skip_patterns: list[str] = []
    target_signals: list[str] = []
    # Pydantic v2 ignores unknown keys by default (no extra="forbid" is set
    # anywhere in this schema), so a stored snapshot from BEFORE this typed
    # model existed -- the free-form {based_in, open_to, hard_no} shape --
    # still validates. It does NOT get migrated automatically: it validates
    # by silently ignoring those keys and falling back to LocationPrefs's
    # all-permissive defaults, which is safe (matches nothing extra, hides
    # nothing) but does mean a profile edit is needed to actually restate
    # those preferences in the new shape. See profile_seed.yaml's comment.
    location_prefs: LocationPrefs = LocationPrefs()
    seniority_floor: str | None = None
    seniority_ceiling: str | None = None


class ProfileSnapshotData(BaseModel):
    """The JSONB shape stored in profile_versions.snapshot -- spec section 4.1."""

    experience: list[ExperienceEntry] = []
    projects: list[ProjectEntry] = []
    skills: Skills = Skills()
    credentials: Credentials = Credentials()
    criteria: Criteria = Criteria()


class ProfileSnapshot(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    changelog: str | None
    snapshot: ProfileSnapshotData


class ProfileVersionSummary(BaseModel):
    """Lightweight version-history row (3.9's Profile tab) -- no snapshot
    body, just enough to list "what changed and when." Full detail for any
    one version is GET /profile/current (always the latest one today;
    there's no per-id detail endpoint since nothing needs one yet)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    changelog: str | None


class CreateProfileVersionRequest(BaseModel):
    """POST /profile (4.4): a full replacement snapshot + a required
    changelog -- unlike create_profile_version's own optional changelog
    param, every version created through this endpoint needs a real note
    (this is a user-facing edit, not scripts/seed_profile.py's initial
    seed)."""

    snapshot: ProfileSnapshotData
    changelog: str


class ChatMessage(BaseModel):
    """One turn in the profile-intake chat (app/services/profile_intake.py).
    role mirrors the Anthropic Messages API's own role values directly --
    this is passed straight through to build that call's messages list."""

    role: Literal["user", "assistant"]
    content: str


class ProfileIntakeMessageOut(BaseModel):
    """One persisted turn -- GET /profile/intake-chat/history's shape,
    for the desktop chat panel to render a resumed conversation."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    created_at: datetime


class ProfileIntakeChatRequest(BaseModel):
    """POST /profile/intake-chat. Just the new turn -- unlike a stateless
    chat API, the conversation itself is persisted server-side
    (ProfileIntakeMessage), so there's no history to resend each call."""

    user_message: str


class ProfileIntakeChatResponse(BaseModel):
    """Exactly one of assistant_message (the model asked a question or
    commented in plain text) or proposed_snapshot (the model is ready to
    suggest a concrete update) is set -- never both, matching
    profile_intake.chat()'s tool_choice="auto" turn semantics."""

    assistant_message: str | None = None
    proposed_snapshot: ProfileSnapshotData | None = None
    changelog: str | None = None
    open_questions: list[str] = []


class ParsedFileResponse(BaseModel):
    """POST /profile/intake-chat/parse-file -- extracted plain text only,
    nothing persisted. Loads into the desktop chat panel's input box for
    the user to review/edit before sending, never auto-sent."""

    text: str
