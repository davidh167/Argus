import re

# Conservative: only kills unambiguous non-fits. Everything else passes through
# to Stage 2. False negatives (missing a good role) are worse than false
# positives (wasting a cheap Haiku call) -- spec section 5.

HARD_SKIP_PATTERNS = [
    r"\bsolutions engineer\b",
    r"\baccount executive\b",
    r"\baccount manager\b",
    r"\bsales engineer\b",
    r"\bdeveloper advocate\b",
    r"\bdevrel\b",
    r"\bsite reliability engineer\b",
    r"\bdata center\b",
]

# "knowledge" alone (the spec's original phrasing) is too generic -- a plain
# \b-bounded match still fires on non-ML titles like "Knowledge Management
# Specialist". Scoped to the specific ML/NLP-adjacent phrasings that
# motivated including it, rather than the bare word. "retrieval" keeps its
# existing bare-word match -- real-world non-ML "retrieval" titles are rare
# on the tech-company job boards this tool actually scans.
FAST_TRACK_PATTERNS = [
    r"\bresearch engineer\b",
    r"\bml engineer\b",
    r"\bnlp engineer\b",
    r"\bai engineer\b",
    r"\bmachine learning\b",
    r"\bknowledge (graph|engineer|base)\b",
    r"\bretrieval\b",
    r"\bllm\b",
    r"\blanguage model\b",
]


def classify_title(title: str) -> tuple[str, str]:
    """Stage 1: title-only rules, free and instant. Runs on every posting create.

    Hard-skip patterns are checked before fast-track patterns, deliberately --
    an edge case like "Sales Engineer, ML Platform" should still skip rather
    than fast-track.

    This is a coarse, title-only signal -- it never sees the actual JD.
    stage1_reason says so explicitly, since a "fast_track"/"skip" here is not
    a confident verdict, just a routing decision for what gets a real read
    (Stage 2 on the JD, or Stage 3's full profile-aware analysis).

    Returns (stage1_result, stage1_reason) where result is one of
    'skip' | 'fast_track' | 'analyze'.
    """
    t = title.lower()
    for pat in HARD_SKIP_PATTERNS:
        if re.search(pat, t):
            return "skip", (
                f"title matched hard-skip pattern: {pat!r} -- weak, title-only "
                "signal; the JD itself was never read"
            )
    for pat in FAST_TRACK_PATTERNS:
        if re.search(pat, t):
            return "fast_track", (
                f"title matched fast-track pattern: {pat!r} -- weak, title-only "
                "signal; Stage 2 (JD content) or a manual Stage 3 analysis is "
                "needed for an actual fit read"
            )
    return "analyze", "no title-level signal, defaulting to analyze"
