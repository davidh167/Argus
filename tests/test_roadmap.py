from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy.orm import Session

from app.db import engine
from app.services import roadmap
from app.services.postings import capture_posting, star_posting
from app.services.profile import create_profile_version
from app.services.stage2 import Stage2UnavailableError


def _fake_embed_texts(texts: list[str]) -> np.ndarray:
    """Deterministic stand-in for embeddings.embed_texts -- market_demand
    tests shouldn't depend on a live Ollama instance. Same text always
    yields the same unit vector (so exact-string clustering/classification
    assertions hold); different texts yield independent random unit
    vectors, whose cosine similarity clusters around 0 -- matching the
    real "unrelated phrases" noise floor measured against nomic-embed-text
    -- safely below EMBEDDING_SIMILARITY_THRESHOLD."""
    dim = 32
    vecs = []
    for text in texts:
        rng = np.random.RandomState(abs(hash(text)) % (2**32))
        v = rng.randn(dim)
        vecs.append(v / np.linalg.norm(v))
    return np.stack(vecs)


@pytest.fixture(autouse=True)
def _mock_embeddings(monkeypatch):
    monkeypatch.setattr(roadmap.embeddings, "embed_texts", _fake_embed_texts)


EMPTY_SNAPSHOT = {
    "experience": [],
    "projects": [],
    "skills": {"core": [], "developing": [], "coursework": []},
    "credentials": {"education": [], "oss_contributions": [], "publications": []},
    "criteria": {
        "skip_patterns": [],
        "target_signals": [],
        "location_prefs": {},
        "seniority_floor": None,
        "seniority_ceiling": None,
    },
}


def snapshot_with_core(*core_skills):
    snapshot = {
        **EMPTY_SNAPSHOT,
        "skills": {"core": list(core_skills), "developing": [], "coursework": []},
    }
    return snapshot


@pytest.fixture
def db():
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    create_profile_version(session, snapshot=EMPTY_SNAPSHOT)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _starred_posting_with_gaps(db, name, url, hard_gaps=(), soft_gaps=()):
    # "ML Engineer" fast-tracks at Stage 1, so capture_posting never attempts
    # a real Stage 2 call here.
    posting, _ = capture_posting(
        db, company_name=name, title="ML Engineer", raw_jd_text="build models", url=url
    )
    star_posting(db, posting.id)
    verdict = posting.verdicts[0]
    verdict.stage3_result = {
        "verdict": "Moderate Fit",
        "seniority_match": "Yes",
        "domain_match": "Yes",
        "stack_overlap": [],
        "hard_gaps": list(hard_gaps),
        "soft_gaps": list(soft_gaps),
        "developing_gaps": [],
        "roadmap_gaps": [],
        "roadmap_gap_caveat": None,
        "fde_flag": False,
        "resume_base": "ml_search.tex",
        "one_liner": "ok",
    }
    db.flush()
    return posting


def test_detect_skill_trends_respects_threshold(db):
    # Identical text always clusters together (cosine similarity 1.0
    # regardless of what else is in the corpus), so this stays robust even
    # against real starred-posting data already in the dev DB.
    p1 = _starred_posting_with_gaps(
        db, "TrendCo1", "https://trendco1.example.com/1", hard_gaps=["Rust"]
    )
    p2 = _starred_posting_with_gaps(
        db, "TrendCo2", "https://trendco2.example.com/1", hard_gaps=["Rust"]
    )
    trends = roadmap.detect_skill_trends(db, threshold=2)
    matching = [t for t in trends if {p1.id, p2.id} <= set(t.posting_ids)]
    assert len(matching) == 1
    assert matching[0].skill_or_language == "Rust"


def test_detect_skill_trends_clusters_similar_not_just_identical_text(db):
    # Different phrasings of the same underlying gap -- this is the actual
    # bug fixed: Stage 3 free-writes gap text fresh every call, so exact
    # match essentially never found real trends in practice.
    p1 = _starred_posting_with_gaps(
        db,
        "TrendCo7",
        "https://trendco7.example.com/1",
        hard_gaps=["No large-scale distributed systems experience at big-tech scale"],
    )
    p2 = _starred_posting_with_gaps(
        db,
        "TrendCo8",
        "https://trendco8.example.com/1",
        hard_gaps=[
            "2+ years distributed systems design experience for large-scale systems required"
        ],
    )
    trends = roadmap.detect_skill_trends(db, threshold=2, similarity_threshold=0.2)
    # Scoped to "did these two cluster together" -- not which exact sentence
    # wins as medoid, since the real dev DB may have its own similar-enough
    # starred-posting gap text that legitimately joins the same cluster
    # (confirmed happening: real postings about the same underlying gap).
    matching = [t for t in trends if {p1.id, p2.id} <= set(t.posting_ids)]
    assert len(matching) == 1
    assert isinstance(matching[0].skill_or_language, str) and matching[0].skill_or_language


def test_detect_skill_trends_excludes_below_threshold(db):
    posting = _starred_posting_with_gaps(
        db, "TrendCo3", "https://trendco3.example.com/1", soft_gaps=["Kubernetes"]
    )
    trends = roadmap.detect_skill_trends(db, threshold=2)
    assert all(posting.id not in t.posting_ids for t in trends)


def test_detect_skill_trends_excludes_core_skills(db):
    create_profile_version(db, snapshot=snapshot_with_core("Go"))
    p1 = _starred_posting_with_gaps(
        db, "TrendCo4", "https://trendco4.example.com/1", hard_gaps=["Go"]
    )
    p2 = _starred_posting_with_gaps(
        db, "TrendCo5", "https://trendco5.example.com/1", hard_gaps=["Go"]
    )
    trends = roadmap.detect_skill_trends(db, threshold=2)
    assert "Go" not in {t.skill_or_language for t in trends}
    assert p1.id and p2.id  # postings exist; just excluded via the core-skill filter


def test_detect_skill_trends_ignores_unstarred_and_unanalyzed_postings(db):
    posting, _ = capture_posting(
        db,
        company_name="TrendCo6",
        title="ML Engineer",
        raw_jd_text="x",
        url="https://trendco6.example.com/1",
    )
    # Not starred, and no stage3_result -- must not raise, and this posting's
    # id must not appear in any trend (scoped, not a whole-table emptiness
    # check -- the real dev DB already has other starred postings with real
    # stage3 results from live verification).
    trends = roadmap.detect_skill_trends(db, threshold=1)
    assert posting.starred is False
    assert all(posting.id not in t.posting_ids for t in trends)


def test_detect_skill_trends_excludes_dismissed_even_if_starred(db):
    from app.services.postings import dismiss_posting

    p1 = _starred_posting_with_gaps(
        db, "TrendCoDismiss1", "https://trendcodismiss1.example.com/1", hard_gaps=["Kotlin"]
    )
    p2 = _starred_posting_with_gaps(
        db, "TrendCoDismiss2", "https://trendcodismiss2.example.com/1", hard_gaps=["Kotlin"]
    )
    dismiss_posting(db, p1.id)  # still starred, but dismissed -- "not interested" wins

    trends = roadmap.detect_skill_trends(db, threshold=2)
    matching = [t for t in trends if p2.id in t.posting_ids]
    assert all(p1.id not in t.posting_ids for t in trends)
    # threshold=2 can't be met with p1 excluded and only p2 contributing.
    assert matching == []


def _unstarred_posting_with_stage3(
    db, name, url, *, verdict_label, hard_gaps=(), application_status=None
):
    posting, _ = capture_posting(
        db, company_name=name, title="ML Engineer", raw_jd_text="build models", url=url
    )
    verdict = posting.verdicts[0]
    verdict.stage3_result = {
        "verdict": verdict_label,
        "seniority_match": "Yes",
        "domain_match": "Yes",
        "stack_overlap": [],
        "hard_gaps": list(hard_gaps),
        "soft_gaps": [],
        "developing_gaps": [],
        "roadmap_gaps": [],
        "roadmap_gap_caveat": None,
        "fde_flag": False,
        "resume_base": "ml_search.tex",
        "one_liner": "ok",
    }
    db.flush()
    if application_status is not None:
        from app.services.applications import update_application_status

        update_application_status(db, posting.id, application_status)
    return posting


def test_detect_skill_trends_default_evidence_includes_active_application_not_starred(db):
    p1 = _unstarred_posting_with_stage3(
        db,
        "EvidenceCo1",
        "https://evidenceco1.example.com/1",
        verdict_label="Weak Fit",
        hard_gaps=["Terraform"],
        application_status="applied",
    )
    p2 = _unstarred_posting_with_stage3(
        db,
        "EvidenceCo2",
        "https://evidenceco2.example.com/1",
        verdict_label="Weak Fit",
        hard_gaps=["Terraform"],
        application_status="screening",
    )
    trends = roadmap.detect_skill_trends(db, threshold=2)
    matching = [t for t in trends if {p1.id, p2.id} <= set(t.posting_ids)]
    assert len(matching) == 1


def test_detect_skill_trends_default_evidence_includes_strong_or_moderate_fit_not_starred(db):
    p1 = _unstarred_posting_with_stage3(
        db,
        "EvidenceCo3",
        "https://evidenceco3.example.com/1",
        verdict_label="Strong Fit",
        hard_gaps=["Zig"],
    )
    p2 = _unstarred_posting_with_stage3(
        db,
        "EvidenceCo4",
        "https://evidenceco4.example.com/1",
        verdict_label="Moderate Fit",
        hard_gaps=["Zig"],
    )
    trends = roadmap.detect_skill_trends(db, threshold=2)
    matching = [t for t in trends if {p1.id, p2.id} <= set(t.posting_ids)]
    assert len(matching) == 1


def test_detect_skill_trends_default_evidence_excludes_weak_fit_with_no_other_signal(db):
    posting = _unstarred_posting_with_stage3(
        db,
        "EvidenceCo5",
        "https://evidenceco5.example.com/1",
        verdict_label="Weak Fit",
        hard_gaps=["Erlang"],
    )
    trends = roadmap.detect_skill_trends(db, threshold=1)
    assert all(posting.id not in t.posting_ids for t in trends)


def test_detect_skill_trends_posting_ids_override_bypasses_default_evidence(db):
    # A Weak-Fit, unstarred, no-application posting would never qualify
    # for the default evidence pool -- but an explicit posting_ids override
    # must still pick it up.
    p1 = _unstarred_posting_with_stage3(
        db,
        "EvidenceCo6",
        "https://evidenceco6.example.com/1",
        verdict_label="Weak Fit",
        hard_gaps=["Elixir"],
    )
    p2 = _unstarred_posting_with_stage3(
        db,
        "EvidenceCo7",
        "https://evidenceco7.example.com/1",
        verdict_label="Weak Fit",
        hard_gaps=["Elixir"],
    )
    trends = roadmap.detect_skill_trends(db, threshold=2, posting_ids=[p1.id, p2.id])
    matching = [t for t in trends if {p1.id, p2.id} <= set(t.posting_ids)]
    assert len(matching) == 1


def test_generate_roadmap_brief_raises_stage2_unavailable_without_api_key(db):
    with patch.object(roadmap.settings, "anthropic_api_key", None):
        with patch("app.services.roadmap.Anthropic") as mock_client_cls:
            with pytest.raises(Stage2UnavailableError):
                roadmap.generate_roadmap_brief(
                    db, "Rust", trigger_type="job_market_gap", trend_evidence=[]
                )
            mock_client_cls.assert_not_called()


FAKE_BRIEF_INPUT = {
    "what_it_is": "A systems programming language.",
    "why_it_matters": "Several target roles list it for perf-critical services.",
    "suggested_scope": "Constrained and incremental -- rewrite one small service.",
    "suggested_stack": None,
    "strengthens_project": None,
    "project_note": None,
}


def test_generate_roadmap_brief_parses_tool_use_and_upserts_not_duplicates(db):
    fake_tool_use_block = MagicMock(type="tool_use", input=dict(FAKE_BRIEF_INPUT))
    fake_response = MagicMock(content=[fake_tool_use_block])
    with patch.object(roadmap.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.roadmap.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
            mock_client_cls.return_value = mock_client

            first = roadmap.generate_roadmap_brief(
                db, "Rust", trigger_type="job_market_gap", trend_evidence=[1, 2]
            )
            second = roadmap.generate_roadmap_brief(
                db, "Rust", trigger_type="job_market_gap", trend_evidence=[1, 2, 3]
            )

    assert first.id == second.id
    assert second.what_it_is == "A systems programming language."
    assert second.trend_evidence == [1, 2, 3]
    assert second.is_new_territory is True
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "generate_roadmap_brief"}
    assert call_kwargs["model"] == roadmap.ROADMAP_MODEL


def test_build_roadmap_user_message_wraps_jd_content_in_delimiter_tags(db):
    posting = _starred_posting_with_gaps(db, "TrendCo7", "https://trendco7.example.com/1")
    from app.schemas.profile import ProfileSnapshotData

    msg = roadmap.build_roadmap_user_message(
        skill_or_language="Rust",
        is_new_territory=True,
        profile=ProfileSnapshotData(),
        evidence_postings=[posting],
    )
    assert msg.count("<job_posting>") == 1
    assert msg.count("</job_posting>") == 1
    start = msg.index("<job_posting>")
    end = msg.index("</job_posting>")
    assert start < msg.index("Title: ML Engineer") < end


def test_update_roadmap_status_stamps_completed_at_only_on_complete(db):
    with patch.object(roadmap.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.roadmap.Anthropic") as mock_client_cls:
            fake_tool_use_block = MagicMock(type="tool_use", input=dict(FAKE_BRIEF_INPUT))
            mock_client = MagicMock()
            mock_client.messages.create.return_value = MagicMock(content=[fake_tool_use_block])
            mock_client_cls.return_value = mock_client
            item = roadmap.generate_roadmap_brief(
                db, "Zig", trigger_type="job_market_gap", trend_evidence=[]
            )

    researching = roadmap.update_roadmap_status(db, item.id, "in_progress")
    assert researching.completed_at is None

    completed = roadmap.update_roadmap_status(db, item.id, "complete")
    assert completed.completed_at is not None


def test_update_roadmap_status_returns_none_for_missing_id(db):
    assert roadmap.update_roadmap_status(db, 999999, "complete") is None


def test_get_in_progress_skills_filters_by_status(db):
    with patch.object(roadmap.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.roadmap.Anthropic") as mock_client_cls:
            fake_tool_use_block = MagicMock(type="tool_use", input=dict(FAKE_BRIEF_INPUT))
            mock_client = MagicMock()
            mock_client.messages.create.return_value = MagicMock(content=[fake_tool_use_block])
            mock_client_cls.return_value = mock_client
            a = roadmap.generate_roadmap_brief(
                db, "Elixir", trigger_type="job_market_gap", trend_evidence=[]
            )
            roadmap.generate_roadmap_brief(
                db, "Nim", trigger_type="job_market_gap", trend_evidence=[]
            )

    roadmap.update_roadmap_status(db, a.id, "in_progress")
    in_progress = roadmap.get_in_progress_skills(db)
    assert "Elixir" in in_progress
    assert "Nim" not in in_progress


def _posting_with_stage3(db, name, url, *, verdict_label, seniority_match="Yes", hard_gaps=()):
    posting, _ = capture_posting(
        db, company_name=name, title="ML Engineer", raw_jd_text="build models", url=url
    )
    verdict = posting.verdicts[0]
    verdict.stage3_result = {
        "verdict": verdict_label,
        "seniority_match": seniority_match,
        "domain_match": "Yes",
        "stack_overlap": [],
        "hard_gaps": list(hard_gaps),
        "soft_gaps": [],
        "developing_gaps": [],
        "roadmap_gaps": [],
        "roadmap_gap_caveat": None,
        "fde_flag": False,
        "resume_base": "ml_search.tex",
        "one_liner": f"{verdict_label} for {name}",
    }
    db.flush()
    return posting


def _all_bucket_ids(view):
    return {
        e.posting_id
        for e in (*view.apply_now, *view.solid_match, *view.stretch, *view.build_toward)
    }


def test_readiness_view_apply_now_strong_fit_no_gaps(db):
    posting = _posting_with_stage3(
        db, "ReadyCo1", "https://readyco1.example.com/1", verdict_label="Strong Fit"
    )
    view = roadmap.readiness_view(db)
    assert posting.id in {e.posting_id for e in view.apply_now}
    assert posting.id not in {e.posting_id for e in view.solid_match}
    assert posting.id not in {e.posting_id for e in view.stretch}
    assert posting.id not in {e.posting_id for e in view.build_toward}


def test_readiness_view_solid_match_strong_fit_with_gaps(db):
    # Regression test: Strong Fit wasn't in this tier's eligible verdict set,
    # so a Strong Fit posting with even one hard gap fell through every
    # bucket and vanished from the view. Two gaps is within SOLID_MATCH_HARD_GAP_CAP.
    posting = _posting_with_stage3(
        db,
        "ReadyCo7",
        "https://readyco7.example.com/1",
        verdict_label="Strong Fit",
        hard_gaps=["Node.js", "Elasticsearch"],
    )
    view = roadmap.readiness_view(db)
    assert posting.id in {e.posting_id for e in view.solid_match}
    assert posting.id not in {e.posting_id for e in view.apply_now}
    assert posting.id not in {e.posting_id for e in view.stretch}
    assert posting.id not in {e.posting_id for e in view.build_toward}


def test_readiness_view_weak_fit_with_few_gaps_is_stretch_not_solid_match(db):
    # Weak Fit is a verdict on the role as a whole (e.g. a generic role that
    # underuses the profile); a short gap list must not promote it to Solid.
    posting = _posting_with_stage3(
        db,
        "ReadyCoWeakFew",
        "https://readycoweakfew.example.com/1",
        verdict_label="Weak Fit",
        hard_gaps=["Erlang"],
    )
    view = roadmap.readiness_view(db)
    assert posting.id in {e.posting_id for e in view.stretch}
    assert posting.id not in {e.posting_id for e in view.solid_match}


def test_readiness_view_stretch_moderate_fit_beyond_solid_match_cap(db):
    posting = _posting_with_stage3(
        db,
        "ReadyCo2",
        "https://readyco2.example.com/1",
        verdict_label="Moderate Fit",
        hard_gaps=["Kubernetes at scale", "gRPC", "Terraform"],  # 3 > SOLID_MATCH_HARD_GAP_CAP (2)
    )
    view = roadmap.readiness_view(db)
    assert posting.id in {e.posting_id for e in view.stretch}
    assert posting.id not in {e.posting_id for e in view.apply_now}
    assert posting.id not in {e.posting_id for e in view.solid_match}


def test_readiness_view_stretch_borderline_seniority_no_longer_bypasses_cap(db):
    # A Borderline-seniority posting no longer bypasses the gap cap --
    # confirmed live before this fix: real Moderate Fit/Borderline postings
    # with up to 7 hard gaps all showed as "Stretch" with equal weight. At
    # exactly STRETCH_HARD_GAP_CAP (5) it should still qualify...
    within_cap = _posting_with_stage3(
        db,
        "ReadyCo3",
        "https://readyco3.example.com/1",
        verdict_label="Weak Fit",
        seniority_match="Borderline",
        hard_gaps=["A", "B", "C", "D", "E"],
    )
    # ...but one gap beyond it should be excluded entirely, same as Skip --
    # no "long shot" catch-all bucket (deliberate).
    beyond_cap = _posting_with_stage3(
        db,
        "ReadyCo8",
        "https://readyco8.example.com/1",
        verdict_label="Moderate Fit",
        seniority_match="Borderline",
        hard_gaps=["A", "B", "C", "D", "E", "F"],
    )
    view = roadmap.readiness_view(db)
    assert within_cap.id in {e.posting_id for e in view.stretch}
    assert beyond_cap.id not in _all_bucket_ids(view)


def test_readiness_view_skip_verdict_excluded_from_every_bucket(db):
    posting = _posting_with_stage3(
        db, "ReadyCo4", "https://readyco4.example.com/1", verdict_label="Skip"
    )
    view = roadmap.readiness_view(db)
    assert posting.id not in _all_bucket_ids(view)


def test_readiness_view_weak_fit_unmatched_gap_excluded(db):
    posting = _posting_with_stage3(
        db,
        "ReadyCo5",
        "https://readyco5.example.com/1",
        verdict_label="Weak Fit",
        seniority_match="No",
        # One gap over STRETCH_HARD_GAP_CAP so this doesn't qualify for
        # Solid match or Stretch either -- isolates the "no roadmap match"
        # exclusion path.
        hard_gaps=[
            "Completely unrelated obscure requirement Zzyzx",
            "Another one Qwerp",
            "A third Flerm",
            "A fourth Blorp",
            "A fifth Plugh",
            "A sixth Xyzzy",
        ],
    )
    view = roadmap.readiness_view(db)
    assert posting.id not in _all_bucket_ids(view)


def test_readiness_view_weak_fit_matches_existing_roadmap_item(db):
    from app.models.skill_roadmap import SkillRoadmap

    roadmap_item = SkillRoadmap(
        skill_or_language="Distributed systems design experience", status="flagged"
    )
    db.add(roadmap_item)
    db.flush()

    posting = _posting_with_stage3(
        db,
        "ReadyCo6",
        "https://readyco6.example.com/1",
        verdict_label="Weak Fit",
        seniority_match="No",
        # One gap over STRETCH_HARD_GAP_CAP so this doesn't qualify for
        # Solid match or Stretch either -- isolates the build_toward
        # roadmap-match path.
        hard_gaps=[
            "No distributed systems design experience at scale",
            "Another unrelated obscure requirement Zzyzx",
            "A third unrelated Qwerp requirement",
            "A fourth unrelated Blorp requirement",
            "A fifth unrelated Plugh requirement",
            "A sixth unrelated Xyzzy requirement",
        ],
    )
    view = roadmap.readiness_view(db)
    matching = [e for e in view.build_toward if e.posting_id == posting.id]
    assert len(matching) == 1
    assert matching[0].roadmap_item_id == roadmap_item.id
    assert matching[0].matched_gap == "No distributed systems design experience at scale"


def _posting_with_key_skills(db, name, url, key_skills):
    posting, _ = capture_posting(
        db, company_name=name, title="ML Engineer", raw_jd_text="build models", url=url
    )
    posting.verdicts[0].stage2_key_skills = list(key_skills)
    db.flush()
    return posting


def test_market_demand_empty_when_no_key_skills_recorded(db):
    _capture = capture_posting(
        db,
        company_name="DemandCoNone",
        title="ML Engineer",
        raw_jd_text="x",
        url="https://demandconone.example.com/1",
    )
    result = roadmap.market_demand(db, posting_ids=[_capture[0].id])
    assert result == []


def test_market_demand_clusters_and_counts_postings(db):
    p1 = _posting_with_key_skills(db, "DemandCo1", "https://demandco1.example.com/1", ["PyTorch"])
    p2 = _posting_with_key_skills(db, "DemandCo2", "https://demandco2.example.com/1", ["PyTorch"])
    result = roadmap.market_demand(db, posting_ids=[p1.id, p2.id])
    matching = [d for d in result if d.skill_or_language == "PyTorch"]
    assert len(matching) == 1
    assert matching[0].posting_count == 2
    assert set(matching[0].posting_ids) == {p1.id, p2.id}


def test_market_demand_classifies_have_developing_gap(db):
    create_profile_version(
        db,
        snapshot={
            **EMPTY_SNAPSHOT,
            "skills": {"core": ["PyTorch"], "developing": ["Kubernetes"], "coursework": []},
        },
    )
    p_have = _posting_with_key_skills(
        db, "DemandCoHave", "https://demandcohave.example.com/1", ["PyTorch"]
    )
    p_dev = _posting_with_key_skills(
        db, "DemandCoDev", "https://demandcodev.example.com/1", ["Kubernetes"]
    )
    p_gap = _posting_with_key_skills(
        db, "DemandCoGap", "https://demandcogap.example.com/1", ["Erlang"]
    )

    result = roadmap.market_demand(db, posting_ids=[p_have.id, p_dev.id, p_gap.id])
    by_posting = {}
    for cluster in result:
        for pid in cluster.posting_ids:
            by_posting[pid] = cluster.status
    assert by_posting[p_have.id] == "have"
    assert by_posting[p_dev.id] == "developing"
    assert by_posting[p_gap.id] == "gap"


def test_market_demand_posting_ids_scopes_the_aggregation(db):
    p1 = _posting_with_key_skills(
        db, "DemandCoScope1", "https://demandcoscope1.example.com/1", ["Zig"]
    )
    p2 = _posting_with_key_skills(
        db, "DemandCoScope2", "https://demandcoscope2.example.com/1", ["Zig"]
    )
    result = roadmap.market_demand(db, posting_ids=[p1.id])
    matching = [d for d in result if d.skill_or_language == "Zig"]
    assert len(matching) == 1
    assert matching[0].posting_ids == [p1.id]
    assert p2.id not in matching[0].posting_ids


def _bulk_posting_with_key_skills(db, name, url, key_skills):
    posting = _posting_with_key_skills(db, name, url, key_skills)
    posting.source = "scraped"  # bulk-ingested, not hand-picked
    db.flush()
    return posting


def test_interest_weights_bulk_is_low_and_manual_capture_or_star_raise_it(db):
    bulk = _bulk_posting_with_key_skills(
        db, "IntCoBulk", "https://intcobulk.example.com/1", ["Zig"]
    )
    manual = _posting_with_key_skills(
        db, "IntCoManual", "https://intcomanual.example.com/1", ["Zig"]
    )
    starred = _bulk_posting_with_key_skills(
        db, "IntCoStar", "https://intcostar.example.com/1", ["Zig"]
    )
    starred.starred = True
    both = _posting_with_key_skills(db, "IntCoBoth", "https://intcoboth.example.com/1", ["Zig"])
    both.starred = True
    db.flush()

    w = roadmap.interest_weights(db, [bulk, manual, starred, both])
    assert w[bulk.id] == 0.0
    assert w[manual.id] == roadmap.INTEREST_WEIGHT_MANUAL_CAPTURE
    assert w[starred.id] == roadmap.INTEREST_WEIGHT_STARRED
    assert w[both.id] == w[manual.id] + roadmap.INTEREST_WEIGHT_STARRED


def test_interest_weights_dismissed_counts_for_nothing(db):
    from datetime import datetime, timezone

    posting = _posting_with_key_skills(
        db, "IntCoDismissed", "https://intcodismissed.example.com/1", ["Zig"]
    )
    posting.starred = True
    posting.dismissed_at = datetime.now(timezone.utc)
    db.flush()
    assert roadmap.interest_weights(db, [posting])[posting.id] == 0.0


def test_interest_weights_active_application_raises_it_but_rejected_does_not(db):
    from app.services.applications import update_application_status

    applied = _bulk_posting_with_key_skills(
        db, "IntCoApplied", "https://intcoapplied.example.com/1", ["Zig"]
    )
    rejected = _bulk_posting_with_key_skills(
        db, "IntCoRejected", "https://intcorejected.example.com/1", ["Zig"]
    )
    update_application_status(db, applied.id, status="applied")
    update_application_status(db, rejected.id, status="rejected")
    w = roadmap.interest_weights(db, [applied, rejected])
    assert w[applied.id] == roadmap.INTEREST_WEIGHT_ACTIVE_APPLICATION
    assert w[rejected.id] == 0.0


def test_market_demand_interest_sort_ranks_hand_picked_skill_above_a_bulk_majority(db):
    # 3 bulk postings want "Rarebulk" (no explicit weight); 1 manual capture
    # wants "Handpicked" (1.0). Raw counts say Rarebulk; interest says Handpicked.
    bulk_ids = [
        _bulk_posting_with_key_skills(
            db, f"IntBulk{i}", f"https://intbulk{i}.example.com/1", ["Rarebulk"]
        ).id
        for i in range(3)
    ]
    hand = _posting_with_key_skills(db, "IntHand", "https://inthand.example.com/1", ["Handpicked"])
    scope = [*bulk_ids, hand.id]

    by_market = [
        d.skill_or_language for d in roadmap.market_demand(db, posting_ids=scope, sort="market")
    ]
    by_interest = roadmap.market_demand(db, posting_ids=scope, sort="interest")
    assert by_market.index("Rarebulk") < by_market.index("Handpicked")
    assert [d.skill_or_language for d in by_interest][:2] == ["Handpicked", "Rarebulk"]

    handpicked = next(d for d in by_interest if d.skill_or_language == "Handpicked")
    rarebulk = next(d for d in by_interest if d.skill_or_language == "Rarebulk")
    assert handpicked.interest_weight == 1.0 and handpicked.interest_posting_count == 1
    assert rarebulk.interest_weight == 0.0 and rarebulk.interest_posting_count == 0


def _controlled_embed(monkeypatch, near_pairs: list[tuple[str, str]]):
    """Fake embeddings where each (a, b) pair is ~0.98 similar and all else is
    unrelated (independent random unit vectors)."""
    base = {}

    def embed(texts):
        out = []
        for t in texts:
            key = t
            for a, b in near_pairs:
                if t == b:
                    key = a
            if key not in base:
                base[key] = _fake_embed_texts([key])[0]
            v = base[key].copy()
            if key != t:
                noise = _fake_embed_texts([t + "!noise"])[0]
                v = v + 0.2 * noise
            out.append(v / np.linalg.norm(v))
        return np.stack(out)

    monkeypatch.setattr(roadmap.embeddings, "embed_texts", embed)


def test_market_demand_collapses_case_and_hyphen_variants(db):
    p1 = _posting_with_key_skills(
        db, "DemandCoCase1", "https://demandcocase1.example.com/1", ["RAG Pipelines"]
    )
    p2 = _posting_with_key_skills(
        db, "DemandCoCase2", "https://demandcocase2.example.com/1", ["rag-pipelines"]
    )
    result = roadmap.market_demand(db, posting_ids=[p1.id, p2.id])
    assert len(result) == 1
    assert result[0].posting_count == 2


def test_market_demand_bulk_phrase_joins_nearby_interest_cluster(db, monkeypatch):
    _controlled_embed(monkeypatch, [("Handpicked skill", "Handpicked skills")])
    hand = _posting_with_key_skills(
        db, "DemandCoJoin1", "https://demandcojoin1.example.com/1", ["Handpicked skill"]
    )
    bulk = _bulk_posting_with_key_skills(
        db, "DemandCoJoin2", "https://demandcojoin2.example.com/1", ["Handpicked skills"]
    )
    result = roadmap.market_demand(db, posting_ids=[hand.id, bulk.id], sort="interest")
    assert len(result) == 1
    assert set(result[0].posting_ids) == {hand.id, bulk.id}
    assert result[0].interest_posting_count == 1  # only the hand-picked one


def test_market_demand_project_stack_counts_as_have(db):
    create_profile_version(
        db,
        snapshot={
            **EMPTY_SNAPSHOT,
            "projects": [{"name": "P", "description": "d", "stack": ["Rustlang"]}],
        },
    )
    p = _posting_with_key_skills(
        db, "DemandCoStack", "https://demandcostack.example.com/1", ["Rustlang"]
    )
    result = roadmap.market_demand(db, posting_ids=[p.id])
    assert result[0].status == "have"
