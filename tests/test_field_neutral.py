"""The pipeline is meant to be field-neutral: Stage 1 title phrases come from the profile, and the
LLM prompts carry no ML wording. These tests run the deterministic paths with a clearly non-tech
profile (a fictional registered nurse) and a mocked Anthropic client, so nothing here calls an API.
"""

import re
from unittest.mock import MagicMock, patch

import pytest

from app.models.posting import Posting
from app.schemas.profile import (
    Credentials,
    Criteria,
    ExperienceEntry,
    LocationPrefs,
    ProfileSnapshotData,
    ProjectEntry,
    Skills,
)
from app.services import profile_intake, relevance, roadmap, skill_judgment, stage2, stage3
from app.services.jd_sections import extract_jd_sections
from app.services.stage1 import classify_title
from app.utils.location import matches_location

NURSE = ProfileSnapshotData(
    experience=[
        ExperienceEntry(
            company="Harbor General Hospital (fictional)",
            title="Registered Nurse, Med-Surg",
            date_range="2019-2024",
            signals=["Cared for 6 patients per shift on a 32-bed unit", "Precepted new graduates"],
            weight="high",
        )
    ],
    projects=[
        ProjectEntry(
            name="Falls reduction huddle",
            description="Introduced a bedside safety huddle that cut unit falls.",
            stack=["root cause analysis", "Epic"],
            key_metrics=["falls down 30%"],
        )
    ],
    skills=Skills(
        core=["medication administration", "wound care", "Epic charting", "BLS"],
        developing=["telemetry monitoring"],
        coursework=["pediatric nursing"],
    ),
    credentials=Credentials(education=["BSN, State University"], publications=[]),
    criteria=Criteria(
        skip_patterns=["travel contracts", "administrative-only roles"],
        target_signals=["acute care nursing", "medical-surgical floor", "patient safety"],
        title_fast_track=["registered nurse", "rn", "charge nurse"],
        title_skip=["recruiter", "billing specialist", "software engineer"],
        location_prefs=LocationPrefs(
            cities=["Portland"],
            exclude_cities=["Anchorage"],
            remote_ok=False,
            hybrid_ok=False,
            onsite_ok=True,
        ),
        seniority_floor="2 years of experience",
        seniority_ceiling="charge nurse",
    ),
)

NURSE_JD = (
    "Position Summary: Provide direct patient care on a medical-surgical floor. "
    "Essential Duties: Administer medications, perform wound care, and chart in Epic. "
    "Education and Experience: BSN preferred; 2 years of acute care nursing. "
    "Preferred Qualifications: Telemetry monitoring experience and BLS certification."
)

# Words that would mean an ML-field prompt leaked into a profile-agnostic path.
ML_TERMS = re.compile(
    r"\b(ml|machine learning|llms?|nlp|pytorch|tensorflow|cuda|rag|embeddings?|neural|"
    r"deep learning|transformers?|fine-tun\w*|retrieval)\b",
    re.IGNORECASE,
)


def leaks(text: str) -> set[str]:
    return {m.group(0).lower() for m in ML_TERMS.finditer(text)}


# --- Stage 1 -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Registered Nurse - Med/Surg, Nights", "fast_track"),
        ("RN, Telemetry", "fast_track"),
        ("Charge Nurse (Float Pool)", "fast_track"),
        ("Travel Nurse Recruiter", "skip"),
        ("Medical Billing Specialist", "skip"),
        ("Software Engineer, Clinical Systems", "skip"),
        # Nothing in the profile's phrases matches, so it is routed on, not guessed at.
        ("Nurse Educator", "analyze"),
        # "rn" must match as a whole word, not inside "Intern" or "Northern".
        ("Northern Region Intern", "analyze"),
    ],
)
def test_stage1_routes_titles_from_the_nurse_profiles_own_phrases(title, expected):
    result, _reason = classify_title(title, NURSE.criteria)
    assert result == expected


def test_stage1_with_no_phrases_routes_nothing():
    # No built-in ML or software defaults: an empty profile skips and fast-tracks nothing.
    for title in ["Machine Learning Engineer", "Account Executive", "Registered Nurse"]:
        assert classify_title(title, Criteria())[0] == "analyze"


def test_stage1_skip_wins_over_fast_track():
    result, reason = classify_title("Registered Nurse Recruiter", NURSE.criteria)
    assert result == "skip"
    assert "recruiter" in reason


# --- Stage 2 / Stage 3 prompts and message builders --------------------------------------------


def _sections():
    return extract_jd_sections(NURSE_JD)


def test_nursing_jd_headers_are_split_into_sections():
    sections = _sections()
    assert "medical-surgical" in sections.summary
    assert "wound care" in sections.responsibilities
    assert "BSN" in sections.required_qualifications
    assert "Telemetry" in sections.preferred_qualifications


def test_stage2_user_message_carries_the_nurse_profile_and_no_ml_wording():
    msg = stage2.build_user_message(
        "Registered Nurse - Med/Surg",
        _sections(),
        NURSE.criteria,
        education=NURSE.credentials.education,
        skills_core=NURSE.skills.core,
        skills_developing=NURSE.skills.developing,
    )
    assert "acute care nursing" in msg
    assert "BSN, State University" in msg
    assert "medication administration" in msg
    assert msg.index("telemetry monitoring") < msg.index("<job_posting>")
    assert leaks(msg) == set()


def test_stage3_user_message_carries_the_nurse_profile_and_no_ml_wording():
    msg = stage3.build_user_message(
        title="Registered Nurse - Med/Surg",
        sections=_sections(),
        profile=NURSE,
        roadmap_items=["ventilator management"],
    )
    for expected in ["Registered Nurse, Med-Surg", "Falls reduction huddle", "wound care"]:
        assert expected in msg
    assert "ventilator management" in msg
    assert leaks(msg) == set()


def test_stage2_triage_sends_nurse_context_to_a_mocked_client():
    tool_use = MagicMock(type="tool_use")
    tool_use.input = {
        "result": "fast_track",
        "reason": "Core med-surg floor nursing.",
        "key_skills": ["medication administration", "BLS"],
    }
    response = MagicMock(
        content=[tool_use],
        stop_reason="tool_use",
        usage=MagicMock(input_tokens=500, output_tokens=60),
    )
    with patch.object(stage2.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage2.Anthropic") as client_cls:
            client_cls.return_value.messages.create.return_value = response
            result, _reason, key_skills, tokens = stage2.triage(
                title="Registered Nurse - Med/Surg",
                sections=_sections(),
                criteria=NURSE.criteria,
                skills_core=NURSE.skills.core,
            )
    kwargs = client_cls.return_value.messages.create.call_args.kwargs
    assert result == "fast_track"
    assert key_skills == ["medication administration", "BLS"]
    assert tokens == 560
    assert "medical-surgical floor" in kwargs["messages"][0]["content"]
    assert kwargs["system"] is stage2.SYSTEM_PROMPT


def test_stage3_analyze_sends_nurse_context_to_a_mocked_client():
    result_input = {
        "verdict": "Strong Fit",
        "seniority_match": "Yes",
        "domain_match": "Yes",
        "stack_overlap": ["wound care"],
        "hard_gaps": [],
        "soft_gaps": [],
        "developing_gaps": ["telemetry monitoring"],
        "roadmap_gaps": [],
        "roadmap_gap_caveat": None,
        "fde_flag": False,
        "resume_base": "med_surg_rn.docx",
        "one_liner": "Direct match on acute care floor nursing.",
    }
    tool_use = MagicMock(type="tool_use", input=result_input)
    response = MagicMock(content=[tool_use], usage=MagicMock(input_tokens=900, output_tokens=150))
    with patch.object(stage3.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.stage3.Anthropic") as client_cls:
            client_cls.return_value.messages.create.return_value = response
            analysis, tokens = stage3.analyze(
                title="Registered Nurse - Med/Surg", sections=_sections(), profile=NURSE
            )
    kwargs = client_cls.return_value.messages.create.call_args.kwargs
    assert analysis.stack_overlap == ["wound care"]
    assert tokens == 1050
    assert "Falls reduction huddle" in kwargs["messages"][0]["content"]


def test_roadmap_and_skill_judgment_messages_render_nurse_data():
    posting = Posting(title="ICU Registered Nurse", raw_jd_text=NURSE_JD)
    roadmap_msg = roadmap.build_roadmap_user_message(
        skill_or_language="telemetry monitoring",
        is_new_territory=False,
        profile=NURSE,
        evidence_postings=[posting],
    )
    assert "Falls reduction huddle" in roadmap_msg
    assert "root cause analysis" in roadmap_msg
    assert "BSN preferred" in roadmap_msg
    assert leaks(roadmap_msg) == set()

    evidence = skill_judgment.evidence_items(NURSE)
    texts = " ".join(t for _id, t in evidence)
    assert "Listed skill (core): BLS" in texts
    assert "Precepted new graduates" in texts
    judgment_msg = skill_judgment.build_user_message(
        "cardiac monitoring", ["telemetry", "cardiac monitoring"], evidence[:3]
    )
    assert "cardiac monitoring" in judgment_msg
    assert leaks(judgment_msg) == set()


@pytest.mark.parametrize(
    "prompt",
    [
        pytest.param(stage3.SYSTEM_PROMPT, id="stage3"),
        pytest.param(roadmap.SYSTEM_PROMPT, id="roadmap"),
        pytest.param(profile_intake.SYSTEM_PROMPT, id="profile_intake"),
    ],
)
def test_system_prompts_have_no_ml_wording(prompt):
    assert leaks(prompt) == set()


def test_stage2_system_prompt_has_no_ml_wording():
    assert leaks(stage2.SYSTEM_PROMPT) == set()


def test_skill_judgment_system_prompt_has_no_ml_wording():
    assert leaks(skill_judgment.SYSTEM_PROMPT) == set()


def test_stage2_key_skills_guidance_keeps_its_cap_and_spans_several_fields():
    prompt = stage2.SYSTEM_PROMPT
    assert "up to 12 skills" in prompt
    assert leaks(prompt) == set()
    for example in ["wound care", "CNC machining", "financial modeling", "distributed systems"]:
        assert example in prompt


# --- Relevance ranking and location -------------------------------------------------------------

POSTINGS = [
    (
        "ml_engineer",
        "Machine Learning Engineer. Train transformer models in PyTorch, build embedding "
        "pipelines and deploy inference services on Kubernetes. Python required.",
    ),
    (
        "med_surg_rn",
        "Registered Nurse, Med-Surg. Acute care nursing on a medical-surgical floor: medication "
        "administration, wound care, Epic charting, patient safety huddles. BLS required.",
    ),
    (
        "billing",
        "Medical Billing Specialist. Submit claims, resolve denials and reconcile payer "
        "remittances. Experience with coding systems.",
    ),
    (
        "icu_rn",
        "ICU Registered Nurse. Telemetry monitoring and acute care patient safety; medication "
        "administration and wound care as needed. BLS and ACLS required.",
    ),
]


def test_relevance_ranks_nursing_postings_above_a_tech_posting_for_a_nurse():
    query = relevance.profile_query_text(NURSE)
    ranked = relevance.rank_by_relevance(list(POSTINGS), query)
    assert set(ranked[:2]) == {"med_surg_rn", "icu_rn"}
    # The ML posting shares nothing with this profile: no built-in tech boost.
    assert ranked.index("ml_engineer") >= 2


def test_relevance_ranking_flips_for_a_tech_profile_on_the_same_postings():
    # Same corpus, ML-leaning query: the ranking follows the profile, not a fixed field.
    ranked = relevance.rank_by_relevance(
        list(POSTINGS), "machine learning pytorch transformer inference python"
    )
    assert ranked[0] == "ml_engineer"


def test_relevance_query_uses_positive_signals_only():
    query = relevance.profile_query_text(NURSE)
    assert "wound care" in query and "acute care nursing" in query
    assert "travel contracts" not in query  # skip_patterns never feed ranking


@pytest.mark.parametrize(
    ("workplace_type", "locations", "expected"),
    [
        ("onsite", ["Portland, OR"], True),
        ("hybrid", ["Portland, OR"], False),  # hybrid_ok is off for this nurse
        ("remote", ["United States"], False),  # remote_ok is off
        ("onsite", ["Seattle, WA"], False),
        ("onsite", ["Anchorage, AK"], False),
        (None, None, None),  # no location data: neither a match nor a rejection
    ],
)
def test_location_logic_for_an_onsite_only_nurse(workplace_type, locations, expected):
    got = matches_location(
        NURSE.criteria.location_prefs, workplace_type=workplace_type, locations=locations
    )
    assert got is expected
