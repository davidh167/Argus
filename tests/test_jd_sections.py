from app.services.jd_sections import (
    MAX_SECTION_CHARS,
    build_normalized_jd_text,
    extract_jd_sections,
)


def test_greenhouse_shaped_single_line_text():
    # Greenhouse's crawler-side HTML stripping collapses all whitespace to
    # single spaces before this ever reaches the extractor.
    text = (
        "About the Role We build search systems for millions of users. "
        "What You'll Do Design pipelines. Build ranking models. Ship to prod. "
        "Minimum Qualifications 2+ years experience. Proficient in Python. "
        "Preferred Qualifications Experience with PyTorch. RESTful services."
    )
    sections = extract_jd_sections(text)
    assert "search systems" in sections.summary
    assert "Design pipelines" in sections.responsibilities
    assert "2+ years experience" in sections.required_qualifications
    assert "PyTorch" in sections.preferred_qualifications


def test_lever_shaped_text_with_real_newlines():
    text = (
        "About This Role\nWe are hiring an ML engineer.\n\n"
        "Responsibilities\n- Build models\n- Ship features\n\n"
        "Requirements\n- 3+ years experience\n\n"
        "Nice to Have\n- Kubernetes experience"
    )
    sections = extract_jd_sections(text)
    assert "ML engineer" in sections.summary
    assert "Build models" in sections.responsibilities
    assert "3+ years" in sections.required_qualifications
    assert "Kubernetes" in sections.preferred_qualifications


def test_manually_pasted_text_with_dashes():
    text = (
        "Join our team building AI products.\n\n"
        "What you'll do:\n- Design systems\n- Write code\n\n"
        "Requirements:\n- BS in CS\n- 2 years experience"
    )
    sections = extract_jd_sections(text)
    assert "AI products" in sections.summary
    assert "Design systems" in sections.responsibilities
    assert "BS in CS" in sections.required_qualifications


def test_no_headers_at_all_falls_back_to_lead_excerpt():
    text = "Just a plain paragraph describing a job with no recognizable section headers at all."
    sections = extract_jd_sections(text)
    assert sections.summary.startswith("Just a plain paragraph")
    assert sections.responsibilities == ""
    assert sections.required_qualifications == ""
    assert sections.preferred_qualifications == ""


def test_empty_text_returns_all_empty_sections():
    sections = extract_jd_sections("")
    assert sections.summary == ""
    assert sections.responsibilities == ""
    assert sections.required_qualifications == ""
    assert sections.preferred_qualifications == ""


def test_section_length_is_capped():
    long_body = "x " * 2000  # far longer than MAX_SECTION_CHARS
    text = f"Minimum Qualifications {long_body} Preferred Qualifications short bonus text"
    sections = extract_jd_sections(text)
    assert len(sections.required_qualifications) <= MAX_SECTION_CHARS


def test_max_chars_none_uncaps_every_section_including_summary():
    long_body = "x " * 2000
    text = (
        f"About the Role {long_body} "
        f"Minimum Qualifications {long_body} "
        f"Preferred Qualifications {long_body}"
    )
    capped = extract_jd_sections(text)
    uncapped = extract_jd_sections(text, max_chars=None)

    assert len(capped.summary) < len(uncapped.summary)
    assert len(capped.required_qualifications) < len(uncapped.required_qualifications)
    assert len(uncapped.required_qualifications) > MAX_SECTION_CHARS


def test_max_chars_none_uncaps_the_no_headers_fallback():
    long_body = "A plain paragraph with no headers at all. " * 50
    uncapped = extract_jd_sections(long_body, max_chars=None)
    capped = extract_jd_sections(long_body)
    assert len(uncapped.summary) > len(capped.summary)
    assert uncapped.summary.strip() == long_body.strip()


def test_build_normalized_jd_text_joins_sections_and_skips_empty_ones():
    text = (
        "About the Role We build search systems. "
        "What You'll Do Design pipelines. "
        "Minimum Qualifications 2+ years experience."
        # no "Preferred Qualifications" section at all
    )
    normalized = build_normalized_jd_text(text)
    assert "search systems" in normalized
    assert "Design pipelines" in normalized
    assert "2+ years experience" in normalized
    # No stray double-blank-line artifact from the missing preferred section.
    assert "\n\n\n" not in normalized


def test_build_normalized_jd_text_excludes_raw_boilerplate_outside_sections():
    # EEO boilerplate that appears before/after the recognized sections
    # (a company-wide paragraph, not part of this specific role's summary)
    # must not leak into normalized text via raw_jd_text.
    text = (
        "EEO BOILERPLATE TEXT REPEATED ON EVERY POSTING. "
        "About the Role We build search systems. "
        "Minimum Qualifications 2+ years experience."
    )
    normalized = build_normalized_jd_text(text)
    assert "EEO BOILERPLATE" not in normalized


def test_build_normalized_jd_text_none_for_empty_or_none():
    assert build_normalized_jd_text(None) is None
    assert build_normalized_jd_text("") is None
