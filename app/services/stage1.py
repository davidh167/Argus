import re

from app.schemas.profile import Criteria

# Conservative by design: Stage 1 reads only the title, so a "skip" should be an unambiguous
# non-fit and everything else passes through to Stage 2. False negatives (missing a good role) are
# worse than false positives (wasting a cheap Haiku call) -- spec section 5. The phrases come from
# the candidate's profile (Criteria.title_skip / title_fast_track), not from this module, because
# what counts as an obvious non-fit depends on the field.


def _matches(phrase: str, title: str) -> bool:
    # (?<!\w)/(?!\w) rather than \b so a phrase that starts or ends with punctuation ("c++") still
    # matches as a whole word.
    return re.search(rf"(?<!\w){re.escape(phrase.strip().lower())}(?!\w)", title) is not None


def classify_title(title: str, criteria: Criteria) -> tuple[str, str]:
    """Stage 1: title-only rules, free and instant. Runs on every posting create.

    Skip phrases are checked before fast-track phrases, deliberately -- a title like "Sales
    Engineer, ML Platform" should still skip rather than fast-track.

    This is a coarse, title-only signal -- it never sees the actual JD. stage1_reason says so
    explicitly, since a "fast_track"/"skip" here is not a confident verdict, just a routing decision
    for what gets a real read (Stage 2 on the JD, or Stage 3's full profile-aware analysis).

    Returns (stage1_result, stage1_reason) where result is one of
    'skip' | 'fast_track' | 'analyze'.
    """
    t = title.lower()
    for phrase in criteria.title_skip:
        if phrase.strip() and _matches(phrase, t):
            return "skip", (
                f"title matched your title_skip phrase {phrase!r} -- weak, title-only "
                "signal; the JD itself was never read"
            )
    for phrase in criteria.title_fast_track:
        if phrase.strip() and _matches(phrase, t):
            return "fast_track", (
                f"title matched your title_fast_track phrase {phrase!r} -- weak, title-only "
                "signal; Stage 2 (JD content) or a manual Stage 3 analysis is "
                "needed for an actual fit read"
            )
    return "analyze", "no title-level signal, defaulting to analyze"
