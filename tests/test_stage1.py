import pytest

from app.schemas.profile import Criteria
from app.services.stage1 import classify_title
from tests.conftest import TITLE_RULES

CRITERIA = Criteria(**TITLE_RULES)


def classify(title):
    return classify_title(title, CRITERIA)


@pytest.mark.parametrize(
    "title",
    [
        "Solutions Engineer",
        "Account Executive, Enterprise",
        "Senior Account Manager",
        "Sales Engineer II",
        "Developer Advocate",
        "DevRel Engineer",
        "Site Reliability Engineer",
        "Data Center Technician",
    ],
)
def test_hard_skip(title):
    result, _ = classify(title)
    assert result == "skip"


@pytest.mark.parametrize(
    "title",
    [
        "Research Engineer, Alignment",
        "ML Engineer",
        "NLP Engineer II",
        "AI Engineer, Applied",
        "Machine Learning Scientist",
        "Knowledge Engineer",
        "Retrieval Systems Engineer",
        "LLM Engineer",
        "Language Model Researcher",
    ],
)
def test_fast_track(title):
    result, _ = classify(title)
    assert result == "fast_track"


@pytest.mark.parametrize(
    "title",
    [
        "Software Engineer II",
        "Backend Engineer",
        "Product Manager",
        # Bare "knowledge" no longer fast-tracks -- this is a real enterprise
        # knowledge-management role, not ML/NLP, despite containing the word.
        "Knowledge Management Specialist",
    ],
)
def test_default_analyze(title):
    result, _ = classify(title)
    assert result == "analyze"


def test_hard_skip_wins_over_fast_track():
    # Hard-skip must be checked first: this title has an ML signal but is
    # fundamentally a sales-adjacent role.
    result, reason = classify("Sales Engineer, ML Platform")
    assert result == "skip"
    assert "sales engineer" in reason


@pytest.mark.parametrize("title", ["ML Engineer", "Solutions Engineer"])
def test_reason_flags_title_only_weak_signal(title):
    # Every Stage 1 verdict is title-only and never reads the JD -- the
    # reason string should say so explicitly, not just report the pattern.
    _, reason = classify(title)
    assert "weak" in reason.lower()


def test_no_phrases_means_everything_is_analyzed():
    # A profile with no title rules (the field-neutral default) never skips or fast-tracks on title.
    for title in ["ML Engineer", "Account Executive", "Software Engineer"]:
        assert classify_title(title, Criteria())[0] == "analyze"


def test_phrases_match_whole_words_case_insensitively():
    criteria = Criteria(title_fast_track=["  Data Engineer "], title_skip=["intern"])
    assert classify_title("Senior DATA ENGINEER", criteria)[0] == "fast_track"
    assert classify_title("Internal Tools Engineer", criteria)[0] == "analyze"  # not "intern"
    assert classify_title("Data Engineering Lead", criteria)[0] == "analyze"


def test_phrase_with_punctuation_matches_as_a_whole_word():
    criteria = Criteria(title_fast_track=["c++"])
    assert classify_title("Senior C++ Engineer", criteria)[0] == "fast_track"
    assert classify_title("ABC++ Engineer", criteria)[0] == "analyze"  # "c++" inside a longer word


def test_blank_phrase_is_ignored():
    assert classify_title("Backend Engineer", Criteria(title_skip=["", "  "]))[0] == "analyze"
