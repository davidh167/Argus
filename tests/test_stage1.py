import pytest

from app.services.stage1 import classify_title


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
    result, _ = classify_title(title)
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
    result, _ = classify_title(title)
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
    result, _ = classify_title(title)
    assert result == "analyze"


def test_hard_skip_wins_over_fast_track():
    # Hard-skip must be checked first: this title has an ML signal but is
    # fundamentally a sales-adjacent role.
    result, reason = classify_title("Sales Engineer, ML Platform")
    assert result == "skip"
    assert "sales engineer" in reason


@pytest.mark.parametrize("title", ["ML Engineer", "Solutions Engineer"])
def test_reason_flags_title_only_weak_signal(title):
    # Every Stage 1 verdict is title-only and never reads the JD -- the
    # reason string should say so explicitly, not just report the pattern.
    _, reason = classify_title(title)
    assert "weak" in reason.lower()
