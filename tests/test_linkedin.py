from app.utils import linkedin

TRACKED_URL = (
    "https://www.linkedin.com/jobs/view/1111111111/"
    "?trk=eml-email_job_alert&refId=abc%3D%3D&trackingId=xyz"
)

# Shape of a real logged-in capture (abbreviated): nav chrome, job header,
# description, company blurb, sidebar of OTHER jobs, footer, language list.
PAGE = """0 notifications
Skip to search
Home
My Network
Jobs
LinkedIn

AI Engineer

Mountain View, CA · 1 hour ago · Over 100 applicants

Promoted by hirer

Hybrid
Full-time
Easy Apply
Tailor my resume
About the job

Build ranking systems. Requires Python and PyTorch.

About the company

LinkedIn is the world's largest professional network.

More jobs

Machine Learning Engineer

ExampleCorp

San Diego, CA

$175K/yr - $235K/yr

See more jobs like this

Looking for talent?

العربية (Arabic)
Deutsch (German)
"""


def test_is_linkedin_job_url():
    assert linkedin.is_linkedin_job_url(TRACKED_URL)
    assert linkedin.is_linkedin_job_url("https://linkedin.com/jobs/view/123/")
    assert not linkedin.is_linkedin_job_url("https://www.linkedin.com/in/someone/")
    assert not linkedin.is_linkedin_job_url("https://jobs.lever.co/acme/123")
    assert not linkedin.is_linkedin_job_url(None)


def test_canonical_url_drops_tracking_params():
    assert linkedin.canonical_url(TRACKED_URL) == "https://www.linkedin.com/jobs/view/1111111111/"


def test_canonical_url_handles_slugged_view_paths():
    url = "https://www.linkedin.com/jobs/view/ai-engineer-at-linkedin-1111111111?trk=x"
    assert linkedin.canonical_url(url) == "https://www.linkedin.com/jobs/view/1111111111/"


def test_clean_capture_keeps_only_the_description():
    cleaned = linkedin.clean_capture(TRACKED_URL, PAGE)
    assert cleaned.text.startswith("About the job")
    assert "Requires Python and PyTorch" in cleaned.text
    # None of the surrounding page leaks into what Stage 2/3 will read.
    for leaked in (
        "ExampleCorp",
        "$175K",
        "Skip to search",
        "Arabic",
        "world's largest professional network",
    ):
        assert leaked not in cleaned.text


def test_clean_capture_parses_location_workplace_and_url():
    cleaned = linkedin.clean_capture(TRACKED_URL, PAGE)
    assert cleaned.location == "Mountain View, CA"
    assert cleaned.workplace_type == "hybrid"
    assert cleaned.url == "https://www.linkedin.com/jobs/view/1111111111/"


def test_clean_capture_without_about_the_job_keeps_text_but_fixes_url():
    cleaned = linkedin.clean_capture(TRACKED_URL, "some unfamiliar layout")
    assert cleaned.text == "some unfamiliar layout"
    assert cleaned.url == "https://www.linkedin.com/jobs/view/1111111111/"
    assert cleaned.location is None


def test_clean_capture_passes_non_linkedin_pages_through_untouched():
    url = "https://jobs.lever.co/acme/123"
    cleaned = linkedin.clean_capture(url, "About the job\nstuff\nAbout the company\nmore")
    assert cleaned.url == url
    assert cleaned.text == "About the job\nstuff\nAbout the company\nmore"


def test_capture_posting_applies_the_linkedin_cleanup():
    from sqlalchemy.orm import Session

    from app.db import engine
    from app.services.postings import capture_posting
    from app.services.profile import create_profile_version
    from tests.test_relevance import EMPTY_SNAPSHOT

    connection = engine.connect()
    transaction = connection.begin()
    db = Session(bind=connection)
    try:
        create_profile_version(db, snapshot=EMPTY_SNAPSHOT)
        posting, _verdict = capture_posting(
            db,
            company_name="LinkedInCaptureTestCo",
            title="AI Engineer",
            raw_jd_text=PAGE,
            url=TRACKED_URL,
        )
        assert posting.url == "https://www.linkedin.com/jobs/view/1111111111/"
        assert posting.locations == ["Mountain View, CA"]
        assert posting.workplace_type == "hybrid"
        assert "ExampleCorp" not in posting.raw_jd_text
    finally:
        db.close()
        transaction.rollback()
        connection.close()


def test_search_results_page_url_uses_current_job_id():
    url = "https://www.linkedin.com/jobs/search-results/?currentJobId=2222222222&keywords=jobs&refId=x"
    assert linkedin.is_linkedin_job_url(url)
    assert linkedin.canonical_url(url) == "https://www.linkedin.com/jobs/view/2222222222/"


def test_non_job_linkedin_pages_without_a_job_id_are_ignored():
    assert not linkedin.is_linkedin_job_url(
        "https://www.linkedin.com/jobs/search-results/?keywords=jobs"
    )


def test_list_page_location_comes_from_the_selected_job_not_a_card():
    page = (
        "Selected, Other Role\nOther Role\nLinkedIn\nAustin, TX (Remote)\nPosted 1 week ago\n"
        "AI Platform Engineer\nMountain View, CA · 1 week ago · Over 100 applicants\n"
        "Hybrid\nFull-time\nAbout the job\nBuild things.\nAbout the company\nx"
    )
    cleaned = linkedin.clean_capture(
        "https://www.linkedin.com/jobs/search-results/?currentJobId=99", page
    )
    assert cleaned.location == "Mountain View, CA"
    assert cleaned.workplace_type == "hybrid"
    assert cleaned.text == "About the job\nBuild things."


def test_location_ignores_other_lines_that_also_use_the_dot_separator():
    page = (
        "AI Platform Engineer\nMountain View, CA · 1 week ago · Over 100 applicants\n"
        "Promoted by hirer · Actively reviewing applicants\nHybrid\nAbout the job\nBuild things."
    )
    cleaned = linkedin.clean_capture("https://www.linkedin.com/jobs/view/99/", page)
    assert cleaned.location == "Mountain View, CA"
