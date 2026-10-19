from typing import Literal

from pydantic import BaseModel


class Stage3Result(BaseModel):
    verdict: Literal["Strong Fit", "Moderate Fit", "Weak Fit", "Skip"]
    seniority_match: Literal["Yes", "Borderline", "No"]
    domain_match: Literal["Yes", "Partial", "No"]
    stack_overlap: list[str]
    hard_gaps: list[str]
    soft_gaps: list[str]
    developing_gaps: list[str]
    roadmap_gaps: list[str]
    roadmap_gap_caveat: str | None
    fde_flag: bool
    resume_base: str
    one_liner: str
