from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy.orm import Session

from app.db import engine
from app.services import roadmap, skill_judgment
from app.services.postings import capture_posting
from app.services.profile import create_profile_version
from app.services.stage2 import Stage2UnavailableError

SNAPSHOT = {
    "experience": [],
    "projects": [],
    "skills": {"core": ["Zig"], "developing": [], "coursework": []},
    "credentials": {"education": [], "oss_contributions": [], "publications": []},
    "criteria": {
        "skip_patterns": [],
        "target_signals": [],
        "location_prefs": {},
        "seniority_floor": None,
        "seniority_ceiling": None,
    },
}


def _fake_embed_texts(texts):
    vecs = []
    for text in texts:
        rng = np.random.RandomState(abs(hash(text)) % (2**32))
        v = rng.randn(32)
        vecs.append(v / np.linalg.norm(v))
    return np.stack(vecs)


@pytest.fixture(autouse=True)
def _mock_embeddings(monkeypatch):
    monkeypatch.setattr(roadmap.embeddings, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(skill_judgment.embeddings, "embed_texts", _fake_embed_texts)


@pytest.fixture
def db():
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    create_profile_version(session, snapshot=SNAPSHOT)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def fake_client(monkeypatch):
    """Patches Anthropic in skill_judgment; every call answers 'have' citing e1
    plus an id that was never shown (which must be dropped)."""
    # Replace the module's settings object only; patching the shared one would
    # make capture_posting (which triages when a key is set) hit the real API.
    monkeypatch.setattr(skill_judgment, "settings", MagicMock(anthropic_api_key="test-key"))
    block = MagicMock(
        type="tool_use",
        input={"status": "have", "evidence_ids": ["e1", "e999"], "reason": "listed"},
    )
    client = MagicMock()
    client.messages.create.return_value = MagicMock(content=[block])
    with patch.object(skill_judgment, "Anthropic", return_value=client):
        yield client


def _posting(db, name, key_skills, manual=True):
    posting, _ = capture_posting(
        db,
        company_name=name,
        title="Engineer",
        raw_jd_text="build things",
        url=f"https://{name.lower()}.example.com/1",
    )
    posting.verdicts[0].stage2_key_skills = list(key_skills)
    if not manual:
        posting.source = "scraped"
    db.flush()
    return posting


def test_preview_counts_pending_without_calling_the_api(db, fake_client):
    p = _posting(db, "JudgePrev", ["Zig"])
    result = skill_judgment.preview(db, posting_ids=[p.id])
    assert result["to_judge"] == 1 and result["already_judged"] == 0
    assert result["estimated_input_tokens"] > 0 and result["estimated_cost_usd"] > 0
    fake_client.messages.create.assert_not_called()


def test_judge_pending_stores_judgment_and_market_demand_prefers_it(db, fake_client):
    p = _posting(db, "JudgeRun", ["Erlang"])  # similarity guess would say gap
    before = roadmap.market_demand(db, posting_ids=[p.id])[0]
    assert before.status == "gap" and not before.judged

    assert skill_judgment.judge_pending(db, posting_ids=[p.id]) == {"judged": 1, "remaining": 0}
    after = roadmap.market_demand(db, posting_ids=[p.id])[0]
    assert after.status == "have" and after.judged and after.judgment_reason == "listed"

    sent = fake_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Skill: Erlang" in sent and "[e1] Listed skill (core): Zig" in sent


def test_judgment_evidence_drops_ids_that_were_never_shown(db, fake_client):
    from sqlalchemy import select

    from app.models.skill_judgment import SkillJudgment

    p = _posting(db, "JudgeEvid", ["Erlang"])
    skill_judgment.judge_pending(db, posting_ids=[p.id])
    row = db.scalar(select(SkillJudgment).where(SkillJudgment.skill_key == "erlang"))
    assert row.evidence == ["Listed skill (core): Zig"]


def test_already_judged_skills_are_not_judged_again(db, fake_client):
    p = _posting(db, "JudgeTwice", ["Erlang"])
    skill_judgment.judge_pending(db, posting_ids=[p.id])
    assert skill_judgment.judge_pending(db, posting_ids=[p.id]) == {"judged": 0, "remaining": 0}
    assert fake_client.messages.create.call_count == 1


def test_a_new_profile_version_invalidates_judgments(db, fake_client):
    p = _posting(db, "JudgeStale", ["Erlang"])
    skill_judgment.judge_pending(db, posting_ids=[p.id])
    create_profile_version(
        db,
        snapshot={
            **SNAPSHOT,
            "skills": {"core": ["Zig", "Elixir"], "developing": [], "coursework": []},
        },
    )
    cluster = roadmap.market_demand(db, posting_ids=[p.id])[0]
    assert not cluster.judged
    assert skill_judgment.preview(db, posting_ids=[p.id])["to_judge"] == 1


def test_only_interest_set_skills_are_judged(db, fake_client):
    bulk = _posting(db, "JudgeBulk", ["Cobol"], manual=False)
    assert skill_judgment.preview(db, posting_ids=[bulk.id])["to_judge"] == 0
    assert skill_judgment.judge_pending(db, posting_ids=[bulk.id]) == {"judged": 0, "remaining": 0}
    fake_client.messages.create.assert_not_called()


def test_judge_pending_requires_an_api_key(db, monkeypatch):
    monkeypatch.setattr(skill_judgment, "settings", MagicMock(anthropic_api_key=""))
    with pytest.raises(Stage2UnavailableError):
        skill_judgment.judge_pending(db, posting_ids=[])
