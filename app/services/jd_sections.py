import re
from dataclasses import dataclass

# Longest/most-specific phrases first isn't required since we take the first
# (leftmost) match per category, but keep phrases distinct across categories
# so a generic one doesn't shadow a more specific one appearing earlier.
SUMMARY_HEADERS = [
    "about the role",
    "about this role",
    "role summary",
    "about the team",
    "position summary",
    "job summary",
]
RESPONSIBILITIES_HEADERS = [
    "what you'll do",
    "what you will do",
    "what you'll be doing",
    "responsibilities",
    "essential duties",
    "job duties",
]
REQUIRED_HEADERS = [
    "minimum qualifications",
    "required qualifications",
    "basic qualifications",
    "requirements",
    "who you are",
    "what you'll need",
    "what we're looking for",
    "education and experience",
]
PREFERRED_HEADERS = [
    "preferred qualifications",
    "nice to have",
    "bonus points",
    "preferred skills",
    "bonus if",
]

MAX_SECTION_CHARS = 1200  # bounds Haiku token usage/cost per posting
MAX_SUMMARY_CHARS = 500
FALLBACK_SUMMARY_CHARS = 400  # used only when no headers were found at all


@dataclass
class JDSections:
    summary: str
    responsibilities: str
    required_qualifications: str
    preferred_qualifications: str


def extract_jd_sections(raw_jd_text: str, max_chars: int | None = MAX_SECTION_CHARS) -> JDSections:
    """Heuristic-only (no LLM fallback in v1). Finds the first occurrence of
    each category's header keywords as a case-insensitive substring search --
    this works even when the source text has no line breaks (Greenhouse's
    crawler-side HTML stripping already collapses list items to plain
    space-separated text) -- then slices text between consecutive header
    positions. Falls back to a length-capped lead excerpt as `summary` when
    no headers are found at all.

    Each section is returned as a capped text blob rather than a strict list
    of N discrete bullets, since bullet delimiters aren't reliably present in
    already-stored raw_jd_text. max_chars (default MAX_SECTION_CHARS)
    approximates "first ~5 bullets" by length instead of by delimiter count
    -- it exists to bound Stage 2/3's LLM token usage per posting, every
    caller of which uses the default. Pass max_chars=None for
    normalized_jd_text: relevance scoring and similar-roles
    matching want the full section text, not a token-budget-capped excerpt
    -- truncating differently per posting would bias the corpus's TF-IDF
    vocabulary toward whichever postings happened to fit within the cap.
    None also uncaps summary's own (smaller) MAX_SUMMARY_CHARS/
    FALLBACK_SUMMARY_CHARS -- those stay at their fixed values for any
    other max_chars, including the default.
    """
    section_cap = max_chars
    summary_cap = MAX_SUMMARY_CHARS if max_chars is not None else None
    fallback_summary_cap = FALLBACK_SUMMARY_CHARS if max_chars is not None else None

    text = re.sub(r"<[^>]+>", " ", raw_jd_text or "")  # harmless no-op on plain text
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return JDSections("", "", "", "")

    text_lower = text.lower()
    categories = {
        "summary": SUMMARY_HEADERS,
        "responsibilities": RESPONSIBILITIES_HEADERS,
        "required": REQUIRED_HEADERS,
        "preferred": PREFERRED_HEADERS,
    }

    hits: list[tuple[int, int, str]] = []
    for category, phrases in categories.items():
        best = None
        for phrase in phrases:
            idx = text_lower.find(phrase)
            if idx != -1 and (best is None or idx < best[0]):
                best = (idx, idx + len(phrase), category)
        if best:
            hits.append(best)
    hits.sort(key=lambda h: h[0])

    sections = {"summary": "", "responsibilities": "", "required": "", "preferred": ""}
    for i, (_start, end, category) in enumerate(hits):
        next_start = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        sections[category] = text[end:next_start].strip()[:section_cap]

    if not sections["summary"]:
        cutoff = hits[0][0] if hits else len(text)
        sections["summary"] = (text[:cutoff].strip() or text)[:summary_cap]
        if not hits:
            sections["summary"] = text[:fallback_summary_cap]

    return JDSections(
        summary=sections["summary"],
        responsibilities=sections["responsibilities"],
        required_qualifications=sections["required"],
        preferred_qualifications=sections["preferred"],
    )


def build_normalized_jd_text(raw_jd_text: str | None) -> str | None:
    """Posting.normalized_jd_text: summary + responsibilities +
    required + preferred, uncapped, joined into one blob. Used by relevance
    scoring and similar-roles matching INSTEAD of raw_jd_text --
    raw_jd_text still carries EEO/benefits/company-boilerplate paragraphs
    that are identical across every posting from the same company, which
    would otherwise dominate any same-company similarity comparison.
    """
    if not raw_jd_text:
        return None
    sections = extract_jd_sections(raw_jd_text, max_chars=None)
    parts = [
        sections.summary,
        sections.responsibilities,
        sections.required_qualifications,
        sections.preferred_qualifications,
    ]
    joined = "\n\n".join(p for p in parts if p)
    return joined or None
