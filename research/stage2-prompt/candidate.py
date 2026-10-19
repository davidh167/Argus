"""Candidate Stage 2 key_skills prompt (v2). Production is untouched: this
swaps ONLY the key_skills paragraph of app.services.stage2.SYSTEM_PROMPT and
the tool's key_skills description, and raises the cap 8 -> 12."""

import copy

from app.services.stage2 import STAGE2_TOOL, SYSTEM_PROMPT

MAX_KEY_SKILLS_V2 = 12

_OLD_PARAGRAPH = SYSTEM_PROMPT[
    SYSTEM_PROMPT.index("Also record key_skills:") : SYSTEM_PROMPT.index(
        "Call the stage2_triage tool"
    )
]

NEW_PARAGRAPH = f"""Also record key_skills: up to {MAX_KEY_SKILLS_V2} skills this posting actually requires or \
prefers, most important first (required before preferred, the role's core skills before peripheral \
ones). These are aggregated across many postings to track which specific skills keep showing up in \
the job market, independent of this posting's fit verdict -- so writing the same skill the same way \
every time matters more than being exhaustive. For each entry:
- One skill per entry. Split slashes and lists into separate entries ("C/C++" -> "C" and "C++"; \
"React/Next.js" -> "React" and "Next.js"). Keep a slash only when it is part of one established \
name ("CI/CD", "TCP/IP", "A/B testing").
- Use the standard short name practitioners use, singular, with no qualifiers: no years of \
experience, no "strong"/"production"/"hands-on", no versions. "Kubernetes", not "production \
Kubernetes experience"; "Python", not "Python 3.11"; "AutoCAD", not "proficiency in AutoCAD".
- Spell a skill identically every time. If practitioners normally write it as an acronym, use the \
acronym ("LLM", "NLP", "SQL", "CUDA", "RAG", "GAAP"); otherwise spell it out. Never put both an \
acronym and its expansion in the list.
- If the posting names a skill that matches an item in the candidate's skills_core or \
skills_developing, use the candidate's spelling for it. Only include a skill the posting itself \
names or clearly requires -- never add one just because the candidate has it.
- Include broad skill areas as well as specific tools ("machine learning", "distributed systems", \
"financial modeling"): a broad area the posting clearly requires counts as a skill. Leave out soft \
skills ("communication", "teamwork") and role titles.
"""

SYSTEM_PROMPT_V2 = SYSTEM_PROMPT.replace(_OLD_PARAGRAPH, NEW_PARAGRAPH)
# Since v2 was applied to production (app/services/stage2.py) this replace is a no-op; the
# experiment itself ran against the v1 prompt (see README).

STAGE2_TOOL_V2 = copy.deepcopy(STAGE2_TOOL)
STAGE2_TOOL_V2["input_schema"]["properties"]["key_skills"]["description"] = (
    f"Up to {MAX_KEY_SKILLS_V2} skills, one per entry, most important first, each written as the "
    "standard short singular name (see system prompt rules)."
)
