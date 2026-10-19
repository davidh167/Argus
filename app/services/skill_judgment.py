"""LLM judgment of whether the current profile demonstrates a skill the job
market asks for (have / developing / gap). Embeddings and BM25 only
RETRIEVE a shortlist of profile evidence; the model decides. Embedding
similarity alone cannot do this: against labeled clusters no cutoff got more
than 15/22 right, while this approach got 22/22
(research/skill-judgment; labels written with LLM assistance and reviewed by the
author, so read that as "viable", not "calibrated"). Prompt wording assumes no field.

Results are cached per (profile version, skill) in skill_judgments and
consumed by roadmap.market_demand. Paid calls are user-triggered only."""

import numpy as np
from anthropic import Anthropic
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.skill_judgment import SkillJudgment
from app.schemas.profile import ProfileSnapshotData
from app.schemas.roadmap import DemandCluster
from app.services import embeddings, relevance, roadmap
from app.services import profile as profile_service
from app.services.pricing import SONNET_SYNC_INPUT_PRICE_PER_MTOK, SONNET_SYNC_OUTPUT_PRICE_PER_MTOK
from app.services.stage2 import Stage2UnavailableError
from app.services.stage3 import STAGE3_MODEL

TOP_K_EMBED = 8
TOP_K_LEXICAL = 4
DEFAULT_LIMIT = 30
ESTIMATED_OUTPUT_TOKENS = 120

JUDGE_TOOL = {
    "name": "record_skill_judgment",
    "description": "Record whether the candidate's profile demonstrates the skill.",
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["have", "developing", "gap"]},
            "evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ids of the evidence items that support the status (empty for gap).",
            },
            "reason": {"type": "string", "description": "One sentence."},
        },
        "required": ["status", "evidence_ids", "reason"],
        "additionalProperties": False,
    },
    "strict": True,
}

SYSTEM_PROMPT = """You decide whether a candidate's profile demonstrates a skill that job postings ask \
for. You are given the skill (a short label plus the phrasings postings use for it) and a numbered list \
of evidence items retrieved from the candidate's profile: listed skills, projects, work experience, \
and open-source contributions. The list is a shortlist of likely matches, not the whole profile.

Statuses:
- have: the evidence shows the candidate has actually used this skill, or work squarely inside it. A \
tool or framework that implies the skill counts (a shipped retrieval system demonstrates "information \
retrieval"; a project built on a named library demonstrates that library). A broad area counts as have \
when the evidence contains concrete work inside it, not merely a mention of the topic.
- developing: the profile marks the skill as in progress, or shows only adjacent or partial exposure \
(related but different tooling, coursework, or a small part of the skill).
- gap: nothing in the evidence demonstrates it, or the only overlap is a shared word or a different \
skill. Two skills that merely sound alike are not a match (e.g. "C++" is not shown by Python work).

Judge only from the evidence given; do not assume skills from an employer's reputation or a job title. \
Cite the ids of the evidence items you relied on. Call the tool exactly once."""


def evidence_items(profile: ProfileSnapshotData) -> list[tuple[str, str]]:
    """(id, text) for everything in the profile that can demonstrate a skill."""
    items: list[str] = []
    items += [f"Listed skill (core): {s}" for s in profile.skills.core]
    items += [f"Listed skill (developing): {s}" for s in profile.skills.developing]
    for p in profile.projects:
        items.append(f"Project {p.name}: {p.description} Stack: {', '.join(p.stack)}.")
    for e in profile.experience:
        items.append(f"Experience, {e.title} at {e.company}: {'; '.join(e.signals)}")
    for o in profile.credentials.oss_contributions:
        items.append(f"Open source, {o.repo}: {o.description}")
    return [(f"e{i + 1}", t) for i, t in enumerate(items)]


def retrieve(
    label: str, phrasings: list[str], evidence: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Union of the top embedding matches (best over the label and phrasings)
    and the top lexical (BM25) matches, in evidence order."""
    query = [label, *phrasings]
    emb = (embeddings.embed_texts(query) @ embeddings.embed_texts([t for _, t in evidence]).T).max(
        axis=0
    )
    lex = np.array(relevance.bm25_scores([t for _, t in evidence], " ".join(query)))
    keep = set(np.argsort(-emb)[:TOP_K_EMBED]) | {
        i for i in np.argsort(-lex)[:TOP_K_LEXICAL] if lex[i] > 0
    }
    return [evidence[i] for i in sorted(keep)]


def build_user_message(label: str, phrasings: list[str], shortlist: list[tuple[str, str]]) -> str:
    phrases = "; ".join(dict.fromkeys(phrasings))
    evidence = "\n".join(f"[{i}] {t}" for i, t in shortlist)
    return f"Skill: {label}\nPhrasings in postings: {phrases}\n\nEvidence from the candidate's profile:\n{evidence}"


def _targets(
    db: Session, limit: int, posting_ids: list[int] | None
) -> tuple[list[DemandCluster], int, ProfileSnapshotData, int]:
    """The up-to-`limit` interest-set skills (most-wanted first) that have no
    cached judgment for the current profile version, plus how many in-scope
    skills are already judged."""
    version = profile_service.get_current_profile(db)
    profile = ProfileSnapshotData.model_validate(version.snapshot)
    clusters = [
        c
        for c in roadmap.market_demand(db, posting_ids=posting_ids, sort="interest")
        if c.interest_posting_count > 0
    ]
    judged = {
        key
        for key in db.scalars(
            select(SkillJudgment.skill_key).where(SkillJudgment.profile_version_id == version.id)
        )
    }
    pending = [c for c in clusters if roadmap._normalize_phrase(c.skill_or_language) not in judged]
    return pending[:limit], len(clusters) - len(pending), profile, version.id


def _messages(
    clusters: list[DemandCluster], profile: ProfileSnapshotData
) -> list[tuple[DemandCluster, str, dict]]:
    evidence = evidence_items(profile)
    out = []
    for c in clusters:
        shortlist = retrieve(c.skill_or_language, c.phrasings, evidence)
        out.append(
            (c, build_user_message(c.skill_or_language, c.phrasings, shortlist), dict(shortlist))
        )
    return out


def preview(db: Session, limit: int = DEFAULT_LIMIT, posting_ids: list[int] | None = None) -> dict:
    """Read-only: what judge_pending would send and roughly what it costs."""
    clusters, already, profile, _ = _targets(db, limit, posting_ids)
    msgs = _messages(clusters, profile)
    in_tokens = sum((len(SYSTEM_PROMPT) + len(m)) / 4 for _, m, _ in msgs)
    cost = (
        in_tokens * SONNET_SYNC_INPUT_PRICE_PER_MTOK
        + len(msgs) * ESTIMATED_OUTPUT_TOKENS * SONNET_SYNC_OUTPUT_PRICE_PER_MTOK
    ) / 1e6
    return {
        "to_judge": len(msgs),
        "already_judged": already,
        "estimated_input_tokens": int(in_tokens),
        "estimated_cost_usd": round(cost, 4),
    }


def judge_pending(
    db: Session, limit: int = DEFAULT_LIMIT, posting_ids: list[int] | None = None
) -> dict:
    """One synchronous Sonnet call per un-judged interest-set skill. Commits
    after EACH judgment (unlike most services, which only flush): these are
    paid calls, and an API error on call 17 must not discard the 16 before it."""
    if not settings.anthropic_api_key:
        raise Stage2UnavailableError("ANTHROPIC_API_KEY not configured")
    clusters, _, profile, version_id = _targets(db, limit, posting_ids)
    client = Anthropic(api_key=settings.anthropic_api_key)
    for cluster, message, shown in _messages(clusters, profile):
        response = client.messages.create(
            model=STAGE3_MODEL,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": message}],
            tools=[JUDGE_TOOL],
            tool_choice={"type": "tool", "name": JUDGE_TOOL["name"]},
        )
        result = next(b for b in response.content if b.type == "tool_use").input
        db.add(
            SkillJudgment(
                profile_version_id=version_id,
                skill_key=roadmap._normalize_phrase(cluster.skill_or_language),
                skill_label=cluster.skill_or_language,
                status=result["status"],
                # A cited id outside the shortlist is dropped rather than trusted.
                evidence=[shown[i] for i in result["evidence_ids"] if i in shown],
                reason=result["reason"],
                model=STAGE3_MODEL,
            )
        )
        db.commit()
    remaining = len(_targets(db, 10**6, posting_ids)[0])
    return {"judged": len(clusters), "remaining": remaining}
