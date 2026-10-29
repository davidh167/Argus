"""Load fictional demo data into a SEPARATE database so Argus can be tried without an API key.

Everything here is invented: the candidate, the companies, the postings (example.com URLs) and the
Stage 2 / Stage 3 results, which are canned dicts written straight onto the verdict rows. No LLM,
crawl or network call is ever made, and the real database is never touched.

Usage:
    PYTHONPATH=. poetry run python scripts/seed_demo.py [--reset]

The target database is derived from DATABASE_URL by swapping the database name for `argus_demo`.
The script refuses to run unless the target name ends in `_demo` and differs from the configured
database. It creates the database if missing, runs `alembic upgrade head` on it, and seeds it.
`--reset` drops and recreates it first. Re-running without `--reset` is safe: every record is looked
up before it is created, so nothing is duplicated.

When it finishes it prints the commands to run the backend against the demo database. Generating an
API token (scripts/generate_api_token.py) is your own step.

The Skills view embeds skill phrases, so it needs your usual embeddings backend (local Ollama or the
native core). Do not click the Stage 2 / Stage 3 / batch buttons on demo data: they would call the
Anthropic API, which needs a key.
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import psycopg
from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alembic.config import Config  # noqa: E402

from alembic import command  # noqa: E402
from app.config import settings  # noqa: E402
from app.models.application import Application  # noqa: E402
from app.models.company import Company  # noqa: E402
from app.models.posting import Posting  # noqa: E402
from app.models.profile_version import ProfileVersion  # noqa: E402
from app.models.skill_roadmap import SkillRoadmap  # noqa: E402
from app.models.verdict import Verdict  # noqa: E402
from app.schemas.profile import ProfileSnapshotData  # noqa: E402
from app.schemas.stage3 import Stage3Result  # noqa: E402
from app.services.applications import update_application_status  # noqa: E402
from app.services.companies import get_or_create_company  # noqa: E402
from app.services.postings import star_posting, upsert_posting, write_stage1_verdict  # noqa: E402
from app.services.profile import (  # noqa: E402
    NoProfileError,
    create_profile_version,
    get_current_profile,
)  # noqa: E402
from app.services.relevance import rescore_all  # noqa: E402

DEMO_DATABASE = "argus_demo"
DEMO_CHANGELOG = "Demo profile (fictional candidate)"
EXTERNAL_ID_PREFIX = "demo-"


# --------------------------------------------------------------------------------------------
# Profile: a fictional backend/data engineer. Deliberately non-ML.
# --------------------------------------------------------------------------------------------

PROFILE = {
    "experience": [
        {
            "company": "Parsley & Pine Software (fictional)",
            "title": "Data Engineer",
            "date_range": "Aug 2021 -- Present",
            "signals": [
                "Built and operated 40+ Airflow DAGs loading 2 TB/day into a Postgres and S3 warehouse",
                "Cut nightly batch runtime from 6h to 90 min by partitioning and incremental dbt models",
                "Designed a FastAPI service exposing curated metrics to 12 internal teams",
                "Introduced data-quality checks that caught schema drift before dashboards broke",
            ],
            "weight": "high",
        },
        {
            "company": "Kestrel Ledger Co. (fictional)",
            "title": "Backend Engineer",
            "date_range": "Jul 2018 -- Jul 2021",
            "signals": [
                "Maintained billing REST APIs in Python and PostgreSQL serving 300 req/s",
                "Migrated a monolithic cron job suite to queued workers with retries and alerting",
                "Wrote the team's first integration test harness, raising coverage from 20% to 70%",
            ],
            "weight": "medium",
        },
    ],
    "projects": [
        {
            "name": "pipeline-lint",
            "description": (
                "An open-source command line tool that validates Airflow DAGs and dbt projects "
                "against a configurable rule set (naming, retries, ownership, freshness checks). "
                "Used in CI by a handful of small teams."
            ),
            "stack": ["Python", "Airflow", "dbt", "GitHub Actions"],
            "key_metrics": ["120 GitHub stars", "rules covered by 200+ unit tests"],
            "visibility": "public",
            "weight": "high",
        },
        {
            "name": "City Transit Lakehouse",
            "description": (
                "A personal project ingesting public transit feeds into a partitioned Parquet "
                "lake on S3, modeled with dbt and served through a small FastAPI app."
            ),
            "stack": ["Python", "S3", "Parquet", "dbt", "FastAPI", "Docker"],
            "key_metrics": ["8 GB of feeds ingested daily", "p95 API latency under 150 ms"],
            "visibility": "deployed",
            "weight": "medium",
        },
    ],
    "skills": {
        "core": [
            "Python",
            "SQL",
            "PostgreSQL",
            "Apache Airflow",
            "dbt",
            "FastAPI",
            "Docker",
            "AWS S3",
            "REST API design",
        ],
        "developing": ["Apache Kafka", "Kubernetes", "Terraform"],
        "coursework": ["Go", "Apache Spark", "Scala"],
    },
    "credentials": {
        "education": ["B.S. Computer Science, Eastfield State University (completed May 2018)"],
        "oss_contributions": [
            {
                "repo": "example-org/pipeline-lint",
                "type": "feature",
                "merged": True,
                "description": "Added a freshness-check rule for dbt sources.",
                "relevance": "Shows hands-on dbt and data-quality tooling.",
            }
        ],
        "publications": [],
    },
    "criteria": {
        "skip_patterns": [
            "requires an active security clearance",
            "mobile app development as the primary duty",
            "front-end only role",
        ],
        "target_signals": [
            "data pipelines",
            "ETL and ELT",
            "data warehouse modeling",
            "backend services and REST APIs",
            "batch and incremental processing",
            "data quality and observability",
        ],
        "title_fast_track": [
            "data engineer",
            "backend engineer",
            "platform engineer",
            "analytics engineer",
            "data platform",
        ],
        "title_skip": [
            "sales engineer",
            "account executive",
            "recruiter",
            "site reliability engineer",
            "director",
            "intern",
        ],
        "location_prefs": {
            "cities": ["Denver", "Austin"],
            "countries": ["United States"],
            "exclude_cities": [],
            "exclude_countries": [],
            "remote_ok": True,
            "hybrid_ok": True,
            "onsite_ok": False,
        },
        "seniority_floor": "mid",
        "seniority_ceiling": "senior",
    },
}


# --------------------------------------------------------------------------------------------
# Companies and postings.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoCompany:
    name: str
    slug: str
    ats_type: str
    notes: str


COMPANIES = [
    DemoCompany("Brightwater Analytics", "brightwater", "greenhouse", "Self-serve BI vendor."),
    DemoCompany("Quillfeather Health", "quillfeather", "lever", "Clinic scheduling software."),
    DemoCompany("Tidewell Logistics", "tidewell", "ashby", "Regional freight routing."),
    DemoCompany("Copperleaf Payments", "copperleaf", "greenhouse", "Small-business payments."),
    DemoCompany("Marrowstone Games", "marrowstone", "lever", "Mid-size mobile games studio."),
    DemoCompany("Halcyon Grid Energy", "halcyon", "ashby", "Utility metering and forecasting."),
    DemoCompany("Ferncliff Education", "ferncliff", "greenhouse", "K-12 learning platform."),
    DemoCompany("Ostrander Robotics", "ostrander", "lever", "Warehouse robots."),
]

SOURCE_BY_ATS = {"greenhouse": "greenhouse_api", "lever": "lever_api", "ashby": "ashby_api"}


def jd(about: str, do: list[str], need: list[str], nice: list[str], extra: str = "") -> str:
    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {i}" for i in items)

    parts = [
        f"About the role\n{about}",
        f"What you'll do\n{bullets(do)}",
        f"What we're looking for\n{bullets(need)}",
        f"Nice to have\n{bullets(nice)}",
    ]
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)


@dataclass
class DemoPosting:
    company: int  # index into COMPANIES
    title: str
    locations: list[str]
    workplace: str
    days_ago: int
    jd: str
    stage1: str  # expected Stage 1 result, checked against what write_stage1_verdict computes
    stage2: tuple[str, str] | None = None  # (result, reason); None = Stage 2 not run
    key_skills: list[str] | None = None
    stage3: dict | None = None
    comp: tuple[int, int] | None = None  # structured (ATS) compensation, USD per year
    manual_capture: bool = False
    department: str | None = None


def s3(
    verdict: str,
    seniority: str,
    domain: str,
    overlap: list[str],
    hard: list[str],
    line: str,
    *,
    soft: list[str] | None = None,
    developing: list[str] | None = None,
    roadmap: list[str] | None = None,
    caveat: str | None = None,
    fde: bool = False,
    resume: str = "data-engineering",
) -> dict:
    return {
        "verdict": verdict,
        "seniority_match": seniority,
        "domain_match": domain,
        "stack_overlap": overlap,
        "hard_gaps": hard,
        "soft_gaps": soft or [],
        "developing_gaps": developing or [],
        "roadmap_gaps": roadmap or [],
        "roadmap_gap_caveat": caveat,
        "fde_flag": fde,
        "resume_base": resume,
        "one_liner": line,
    }


POSTINGS = [
    # --- Brightwater Analytics -------------------------------------------------------------
    DemoPosting(
        0,
        "Senior Data Engineer",
        ["Remote - United States"],
        "remote",
        6,
        jd(
            "Brightwater's data platform team owns the pipelines behind every customer dashboard. "
            "You will own ingestion and modeling for our largest event sources.",
            [
                "Design and operate Airflow DAGs that load customer event data into Postgres and S3",
                "Model warehouse tables with dbt and keep them tested and documented",
                "Build FastAPI services that expose curated metrics to the product",
                "Own data-quality monitoring and incident follow-up",
            ],
            [
                "5+ years building production data pipelines in Python and SQL",
                "Deep PostgreSQL knowledge, including partitioning and query tuning",
                "Experience with Airflow and dbt",
            ],
            ["Docker", "AWS S3 or similar object storage"],
            "Compensation: $165,000 - $195,000 USD plus equity.",
        ),
        "fast_track",
        key_skills=["Python", "SQL", "PostgreSQL", "Apache Airflow", "dbt", "FastAPI", "AWS S3"],
        stage3=s3(
            "Strong Fit",
            "Yes",
            "Yes",
            ["Python", "SQL", "PostgreSQL", "Apache Airflow", "dbt", "FastAPI", "AWS S3"],
            [],
            "A near-exact match for the candidate's core stack and day-to-day work.",
            soft=["Role asks for 5+ years; candidate has about 5 in data engineering"],
        ),
        department="Data Platform",
    ),
    DemoPosting(
        0,
        "Analytics Engineer",
        ["Denver, CO"],
        "hybrid",
        11,
        jd(
            "Turn raw product events into trusted, well-modeled tables that analysts can use "
            "without asking engineering for help.",
            [
                "Own the dbt project: models, tests, documentation and CI",
                "Partner with analysts to define metrics and keep definitions consistent",
                "Tune slow models and reduce warehouse cost",
            ],
            [
                "3+ years of SQL and dimensional modeling",
                "Strong dbt experience",
                "Comfortable with Python for glue code and tooling",
            ],
            ["Airflow", "Looker or a similar BI tool"],
            "Salary range: $135,000 - $160,000 USD.",
        ),
        "fast_track",
        key_skills=["dbt", "SQL", "Dimensional modeling", "Python", "Apache Airflow", "Looker"],
        stage3=s3(
            "Moderate Fit",
            "Yes",
            "Yes",
            ["dbt", "SQL", "Python", "Apache Airflow"],
            [],
            "Strong dbt and SQL overlap; the role is slightly more analyst-facing than the "
            "candidate's engineering focus.",
            soft=["No Looker experience"],
        ),
        department="Data Platform",
    ),
    DemoPosting(
        0,
        "Sales Engineer, Enterprise",
        ["New York, NY"],
        "hybrid",
        20,
        jd(
            "Join the sales team as the technical voice in enterprise deals.",
            ["Run product demos and proof-of-concept builds", "Answer security questionnaires"],
            ["3+ years in a customer-facing technical role", "Comfort presenting to executives"],
            ["Experience with BI tools"],
        ),
        "skip",
        department="Sales",
    ),
    # --- Quillfeather Health ---------------------------------------------------------------
    DemoPosting(
        1,
        "Backend Engineer, Patient Scheduling",
        ["Remote - United States"],
        "remote",
        3,
        jd(
            "Quillfeather's scheduling engine books millions of clinic visits a year. We need "
            "a backend engineer to extend it to multi-provider and multi-site practices.",
            [
                "Build and evolve REST APIs in Python (FastAPI) backed by PostgreSQL",
                "Integrate with practice-management systems over HL7 and FHIR",
                "Improve reliability with better retries, idempotency and observability",
            ],
            [
                "4+ years of backend development in Python",
                "Strong PostgreSQL and API design skills",
                "Experience integrating third-party systems",
            ],
            ["HL7 or FHIR experience", "Healthcare data experience"],
        ),
        "fast_track",
        key_skills=["Python", "FastAPI", "PostgreSQL", "REST API design", "HL7/FHIR"],
        stage3=s3(
            "Strong Fit",
            "Borderline",
            "Yes",
            ["Python", "FastAPI", "PostgreSQL", "REST API design"],
            ["HL7/FHIR integration experience"],
            "Core backend stack matches closely; the healthcare interchange standards are the "
            "one real gap.",
            soft=["Regulated-data handling (HIPAA) not demonstrated"],
            resume="backend",
        ),
        comp=(150_000, 180_000),
        manual_capture=True,
        department="Engineering",
    ),
    DemoPosting(
        1,
        "Staff Data Platform Engineer",
        ["Austin, TX"],
        "hybrid",
        9,
        jd(
            "Lead the technical direction of Quillfeather's data platform across three teams.",
            [
                "Set architecture standards for ingestion, storage and serving",
                "Mentor senior engineers and review cross-team designs",
                "Drive a multi-quarter migration to a lakehouse architecture",
            ],
            [
                "9+ years of experience, including leading platform-wide initiatives",
                "Expert-level Python and SQL",
                "Track record of scaling data systems across teams",
            ],
            ["Apache Spark", "Terraform"],
        ),
        "fast_track",
        key_skills=["Python", "SQL", "Apache Spark", "Terraform", "Lakehouse architecture"],
        stage3=s3(
            "Moderate Fit",
            "No",
            "Yes",
            ["Python", "SQL", "PostgreSQL"],
            ["9+ years and org-wide technical leadership", "Production Apache Spark"],
            "Domain fits, but the Staff-level scope is well past the candidate's current range.",
            developing=["Terraform"],
        ),
        department="Data Platform",
    ),
    DemoPosting(
        1,
        "Clinical Data Analyst",
        ["Austin, TX"],
        "onsite",
        14,
        jd(
            "Analyze clinic outcomes and prepare monthly reports for partner practices.",
            ["Write SQL against the reporting warehouse", "Build slide decks for partner reviews"],
            ["2+ years as an analyst, ideally in healthcare", "Excellent communication skills"],
            ["Tableau"],
        ),
        "analyze",
        stage2=(
            "skip",
            "Reporting and presentation role; no pipeline or backend engineering work.",
        ),
        key_skills=["SQL", "Tableau", "Healthcare analytics"],
        department="Analytics",
    ),
    # --- Tidewell Logistics ----------------------------------------------------------------
    DemoPosting(
        2,
        "Platform Engineer, Routing",
        ["Austin, TX", "Remote - United States"],
        "hybrid",
        5,
        jd(
            "Tidewell's routing platform plans 80,000 deliveries a day. The platform team keeps "
            "the services, queues and data stores behind it healthy and fast.",
            [
                "Build internal tooling and shared libraries in Python and Go",
                "Operate PostgreSQL and the job queue that feeds the route solver",
                "Improve deploys, observability and cost across Kubernetes workloads",
            ],
            [
                "4+ years in backend or platform engineering",
                "Production experience with Docker and Kubernetes",
                "Strong PostgreSQL skills",
            ],
            ["Go", "Terraform"],
            "Pay band: $150,000 - $178,000 USD.",
        ),
        "fast_track",
        key_skills=["Python", "Go", "PostgreSQL", "Docker", "Kubernetes", "Terraform"],
        stage3=s3(
            "Moderate Fit",
            "Borderline",
            "Partial",
            ["Python", "PostgreSQL", "Docker"],
            ["Production Kubernetes operations", "Go in production"],
            "Python and Postgres line up; the Kubernetes-heavy platform work is still a stretch "
            "for someone only developing those skills.",
            developing=["Kubernetes", "Terraform"],
            resume="backend",
        ),
        department="Platform",
    ),
    DemoPosting(
        2,
        "Software Engineer, Fleet Integrations",
        ["Denver, CO", "Remote - United States"],
        "hybrid",
        8,
        jd(
            "Connect Tidewell to the telematics and ERP systems our customers already run.",
            [
                "Build and maintain integration services and webhooks",
                "Work directly with customer engineers to scope and debug integrations",
                "Write adapters in Python and TypeScript",
            ],
            [
                "3+ years of backend experience",
                "Comfort with REST, webhooks and message queues",
                "Willingness to do customer-facing technical work",
            ],
            ["EDI or SOAP experience", "Apache Kafka"],
        ),
        "analyze",
        stage2=("analyze", "Backend integration work in Python; worth a full read."),
        key_skills=["Python", "TypeScript", "REST API design", "Webhooks", "EDI", "Apache Kafka"],
        stage3=s3(
            "Moderate Fit",
            "Yes",
            "Partial",
            ["Python", "REST API design"],
            ["TypeScript in production", "EDI/SOAP integrations", "Customer-facing consulting"],
            "Backend skills carry over, but the customer-embedded integration focus and TypeScript "
            "requirement make it a stretch.",
            developing=["Apache Kafka"],
            fde=True,
            resume="backend",
        ),
        comp=(140_000, 165_000),
        department="Integrations",
    ),
    DemoPosting(
        2,
        "Site Reliability Engineer",
        ["Remote - United States"],
        "remote",
        18,
        jd(
            "Carry the pager for Tidewell's production routing systems.",
            ["Participate in on-call rotation", "Own SLOs, dashboards and incident reviews"],
            ["4+ years in SRE or operations", "Deep Linux and networking knowledge"],
            ["Prometheus"],
        ),
        "skip",
        department="Reliability",
    ),
    # --- Copperleaf Payments ---------------------------------------------------------------
    DemoPosting(
        3,
        "Backend Engineer, Ledger",
        ["Denver, CO"],
        "hybrid",
        7,
        jd(
            "Copperleaf's ledger records every payment we process. Correctness is the product.",
            [
                "Design double-entry ledger services in Kotlin on the JVM",
                "Own reconciliation jobs and settlement files",
                "Work within PCI-DSS controls on data access and logging",
            ],
            [
                "4+ years of backend engineering, ideally in payments or fintech",
                "Strong Java or Kotlin",
                "Rigorous approach to transactions and consistency",
            ],
            ["PostgreSQL", "Event sourcing"],
            "Salary range: $155,000 - $185,000 USD.",
        ),
        "fast_track",
        key_skills=["Kotlin", "Java", "PostgreSQL", "Double-entry ledgers", "PCI-DSS"],
        stage3=s3(
            "Weak Fit",
            "Yes",
            "Partial",
            ["PostgreSQL"],
            [
                "Kotlin or Java in production",
                "Double-entry ledger design",
                "PCI-DSS compliance work",
                "JVM performance tuning",
            ],
            "Only the database layer overlaps; the JVM language and payments-ledger domain are "
            "both new.",
            soft=["Reconciliation experience is adjacent, not direct"],
            resume="backend",
        ),
        department="Core Payments",
    ),
    DemoPosting(
        3,
        "Data Engineer, Risk",
        ["Remote - United States"],
        "remote",
        4,
        jd(
            "The risk team scores every transaction. Reliable, timely features are what keep "
            "fraud losses down.",
            [
                "Build batch and incremental pipelines that feed risk models",
                "Maintain feature tables in Postgres and S3",
                "Add data-quality checks and lineage",
            ],
            [
                "3+ years building data pipelines in Python and SQL",
                "Experience orchestrating workflows with Airflow",
                "Strong data modeling fundamentals",
            ],
            ["Apache Spark", "dbt"],
            "Compensation: $145,000 - $170,000 USD.",
        ),
        "fast_track",
        key_skills=["Python", "SQL", "Apache Airflow", "Apache Spark", "dbt", "PostgreSQL"],
        stage3=s3(
            "Strong Fit",
            "Yes",
            "Yes",
            ["Python", "SQL", "Apache Airflow", "dbt", "PostgreSQL"],
            ["Production Apache Spark"],
            "Excellent pipeline and Airflow overlap; Spark is a nice-to-have the candidate has "
            "only studied.",
            soft=["Fraud and risk domain knowledge"],
        ),
        department="Risk",
    ),
    DemoPosting(
        3,
        "Director of Engineering, Payments",
        ["Denver, CO"],
        "hybrid",
        25,
        jd(
            "Lead a 30-person engineering organization.",
            ["Own hiring, budget and roadmap", "Report to the CTO"],
            ["10+ years including 5 managing managers"],
            [],
        ),
        "skip",
        department="Engineering",
    ),
    # --- Marrowstone Games -----------------------------------------------------------------
    DemoPosting(
        4,
        "Gameplay Programmer",
        ["Austin, TX"],
        "onsite",
        13,
        jd(
            "Build core gameplay systems for our next mobile title in Unity.",
            ["Implement features in C#", "Profile and optimize frame time on mid-range phones"],
            ["3+ years shipping Unity games", "Strong C#"],
            ["Experience with live-ops tooling"],
        ),
        "analyze",
        stage2=("skip", "Client-side game development in Unity and C#; unrelated to the profile."),
        key_skills=["C#", "Unity", "Mobile optimization"],
        department="Game Engineering",
    ),
    DemoPosting(
        4,
        "Data Engineer, Live Ops Telemetry",
        ["Remote - United States"],
        "remote",
        10,
        jd(
            "Marrowstone's live games emit billions of events a week. Build the streaming "
            "pipeline that powers real-time live-ops decisions.",
            [
                "Operate Kafka topics and Flink jobs for real-time aggregation",
                "Write high-throughput consumers in Go",
                "Serve low-latency metrics from ClickHouse",
                "Run the platform on Kubernetes with Terraform",
            ],
            [
                "4+ years in data engineering with streaming systems in production",
                "Apache Kafka and Apache Flink",
                "Go or Java",
                "ClickHouse or a comparable OLAP store",
            ],
            ["Experience with game telemetry"],
            "Compensation: $150,000 - $175,000 USD.",
        ),
        "fast_track",
        key_skills=[
            "Apache Kafka",
            "Apache Flink",
            "Go",
            "ClickHouse",
            "Kubernetes",
            "Terraform",
            "Python",
        ],
        stage3=s3(
            "Weak Fit",
            "Borderline",
            "Partial",
            ["Python"],
            [
                "Apache Kafka event streaming at scale in production",
                "Apache Flink stateful stream processing",
                "Go in production",
                "ClickHouse or comparable OLAP store",
                "Kubernetes operations in production",
                "Terraform-managed infrastructure",
            ],
            "A streaming-first role; the candidate's batch-pipeline strengths barely apply and "
            "the whole real-time stack is new.",
            developing=["Apache Kafka", "Kubernetes", "Terraform"],
            roadmap=["Apache Kafka"],
            caveat="Kafka is on the roadmap but not yet demonstrated, so it is counted as a gap.",
        ),
        department="Data",
    ),
    DemoPosting(
        4,
        "Technical Recruiter",
        ["Remote - United States"],
        "remote",
        22,
        jd(
            "Recruit engineers and designers for a growing studio.",
            ["Source and screen candidates", "Coordinate interview loops"],
            ["2+ years of technical recruiting"],
            [],
        ),
        "skip",
        department="People",
    ),
    # --- Halcyon Grid Energy ---------------------------------------------------------------
    DemoPosting(
        5,
        "Senior Backend Engineer, Metering",
        ["Remote - United States"],
        "remote",
        2,
        jd(
            "Halcyon turns smart-meter readings into bills and demand forecasts for utilities. "
            "The metering service is the heart of it.",
            [
                "Own the Python services that ingest and validate meter reads",
                "Design PostgreSQL schemas for time-partitioned reading data",
                "Build batch jobs with Airflow for billing-period close",
                "Raise reliability and test coverage across the service",
            ],
            [
                "5+ years of backend engineering in Python",
                "Strong PostgreSQL, including partitioning",
                "Experience with workflow orchestration (Airflow or similar)",
            ],
            ["FastAPI", "Docker", "Utility or IoT data experience"],
            "Compensation: $160,000 - $190,000 USD.",
        ),
        "fast_track",
        key_skills=["Python", "PostgreSQL", "Apache Airflow", "FastAPI", "Docker", "Partitioning"],
        stage3=s3(
            "Strong Fit",
            "Yes",
            "Yes",
            ["Python", "PostgreSQL", "Apache Airflow", "FastAPI", "Docker"],
            [],
            "Nearly a one-to-one match on stack and responsibilities, fully remote.",
            soft=["No direct utility or IoT domain experience"],
            resume="backend",
        ),
        manual_capture=True,
        department="Metering",
    ),
    DemoPosting(
        5,
        "Data Engineer, Grid Forecasting",
        ["Denver, CO"],
        "hybrid",
        12,
        jd(
            "Support Halcyon's forecasting scientists with fast, reliable data foundations.",
            [
                "Build pipelines that assemble weather, load and asset data",
                "Manage time-series storage in TimescaleDB and S3",
                "Productionize forecasting jobs on Airflow",
            ],
            [
                "3+ years in data engineering",
                "Python and SQL",
                "Experience with time-series databases",
                "Exposure to Apache Spark for large backfills",
            ],
            ["Terraform"],
        ),
        "fast_track",
        key_skills=["Python", "SQL", "TimescaleDB", "Apache Airflow", "Apache Spark"],
        stage3=s3(
            "Moderate Fit",
            "Borderline",
            "Yes",
            ["Python", "SQL", "Apache Airflow", "PostgreSQL"],
            [
                "TimescaleDB or other time-series database in production",
                "Apache Spark for large backfills",
                "Weather and load forecasting data",
            ],
            "Pipeline skills transfer cleanly; the time-series storage and Spark expectations are "
            "gaps worth closing.",
            developing=["Terraform"],
        ),
        comp=(130_000, 155_000),
        department="Forecasting",
    ),
    # --- Ferncliff Education ---------------------------------------------------------------
    DemoPosting(
        6,
        "Software Engineer, Learning Platform",
        ["Remote - United States"],
        "remote",
        1,
        jd(
            "Ferncliff serves 2 million students. The learning platform team builds the APIs "
            "teachers and students use every day.",
            [
                "Build Python and PostgreSQL services behind the classroom apps",
                "Improve API performance and reliability",
                "Work with product and design on new assignment workflows",
            ],
            ["3+ years of backend experience", "Python, REST APIs, SQL"],
            ["Experience in education technology"],
            "Salary range: $125,000 - $150,000 USD.",
        ),
        "analyze",
        stage2=("analyze", "Python and PostgreSQL backend role; relevant to the profile."),
        key_skills=["Python", "PostgreSQL", "REST API design", "SQL"],
        department="Learning Platform",
    ),
    DemoPosting(
        6,
        "Analytics Engineer",
        ["Denver, CO"],
        "hybrid",
        3,
        jd(
            "Build the metrics layer that district administrators rely on.",
            ["Model usage data with dbt", "Own dashboards' source tables", "Write data tests"],
            ["2+ years with SQL and dbt", "Clear communication with non-engineers"],
            ["Python", "Airflow"],
        ),
        "fast_track",
        department="Data",
    ),
    # --- Ostrander Robotics ----------------------------------------------------------------
    DemoPosting(
        7,
        "Controls Engineer",
        ["Austin, TX"],
        "onsite",
        15,
        jd(
            "Tune motion controllers for our warehouse robots.",
            ["Develop control loops in C++", "Test on the lab floor"],
            ["MS in controls or robotics", "C++ and MATLAB"],
            ["ROS 2"],
        ),
        "analyze",
        stage2=("skip", "Embedded controls and robotics hardware; no overlap."),
        key_skills=["C++", "MATLAB", "ROS 2", "Control theory"],
        department="Robotics",
    ),
    DemoPosting(
        7,
        "Backend Engineer, Fleet Cloud",
        ["Austin, TX"],
        "onsite",
        6,
        jd(
            "Build the cloud services that coordinate hundreds of robots per warehouse.",
            [
                "Write real-time fleet coordination services in C++ and Rust",
                "Work with ROS 2 message interfaces",
                "Own latency budgets for command dispatch",
            ],
            [
                "5+ years of backend or systems engineering",
                "C++ or Rust in production",
                "Real-time or low-latency systems experience",
            ],
            ["ROS 2", "Kubernetes"],
        ),
        "fast_track",
        key_skills=["C++", "Rust", "ROS 2", "Real-time systems", "Kubernetes"],
        stage3=s3(
            "Skip",
            "Borderline",
            "No",
            [],
            [
                "C++ or Rust in production",
                "Real-time systems",
                "ROS 2",
                "On-site only; candidate prefers remote or hybrid",
            ],
            "Systems-level robotics work in languages the candidate does not use; skip.",
            developing=["Kubernetes"],
            resume="backend",
        ),
        department="Cloud",
    ),
]

# starred posting index -> note
STARRED = {
    0: "Top choice: matches the Airflow and dbt work almost exactly.",
    3: "Check whether HL7/FHIR can be learned on the job.",
    10: "Good Postgres-heavy role; ask about Spark expectations.",
    15: "Fully remote and nearly a perfect stack match.",
}

# posting index -> (status, days since applied or None, follow-up in N days, resume, notes)
APPLICATIONS = {
    0: (
        "applied",
        9,
        5,
        "data-engineering",
        "Applied through the careers page; recruiter replied.",
    ),
    15: ("interview", 14, 2, "backend", "Second round with the metering team lead."),
    10: (
        "screening",
        4,
        7,
        "data-engineering",
        "Recruiter screen done; waiting on a technical call.",
    ),
}


# --------------------------------------------------------------------------------------------
# seed()
# --------------------------------------------------------------------------------------------


def _count(db: Session, model, *where) -> int:
    return db.scalar(select(func.count()).select_from(model).where(*where))


def seed(db: Session) -> dict:
    """Write the demo dataset through the existing services. Flushes but does not commit.
    Idempotent: companies, postings, applications and the roadmap item are looked up before they
    are created, and a posting's verdict is only written when the posting is new."""
    today = date.today()
    now = datetime.now(timezone.utc)

    snapshot = ProfileSnapshotData.model_validate(PROFILE).model_dump(mode="json")
    try:
        current = get_current_profile(db)
    except NoProfileError:
        current = None
    if current is None or current.changelog != DEMO_CHANGELOG:
        current = create_profile_version(db, snapshot=snapshot, changelog=DEMO_CHANGELOG)
    profile_version_id = current.id

    companies: list[Company] = []
    for c in COMPANIES:
        careers_url = f"https://careers.{c.slug}.example.com/"
        company = get_or_create_company(db, c.name, careers_url)
        company.ats_type = c.ats_type
        company.careers_url = careers_url
        # A crawl would try to fetch example.com, so keep the demo companies out of refresh-all.
        company.crawl_enabled = False
        company.notes = c.notes
        companies.append(company)

    postings: list[Posting] = []
    for i, spec in enumerate(POSTINGS):
        company_spec = COMPANIES[spec.company]
        company = companies[spec.company]
        posting, created, _, _ = upsert_posting(
            db,
            company,
            title=spec.title,
            url=f"https://careers.{company_spec.slug}.example.com/jobs/{1000 + i}",
            external_id=f"{EXTERNAL_ID_PREFIX}{i + 1:02d}",
            raw_jd_text=spec.jd,
            posted_date=today - timedelta(days=spec.days_ago),
            source="manual_capture"
            if spec.manual_capture
            else SOURCE_BY_ATS[company_spec.ats_type],
            locations=spec.locations,
            workplace_type=spec.workplace,
            department=spec.department,
            comp_min=spec.comp[0] if spec.comp else None,
            comp_max=spec.comp[1] if spec.comp else None,
            comp_currency="USD" if spec.comp else None,
            comp_period="year" if spec.comp else None,
        )
        postings.append(posting)
        if not created:
            continue

        verdict = write_stage1_verdict(db, posting)
        if verdict.stage1_result != spec.stage1:
            raise RuntimeError(
                f"demo posting {spec.title!r}: expected Stage 1 {spec.stage1!r}, "
                f"got {verdict.stage1_result!r}"
            )
        if spec.stage2 is not None:
            verdict.stage2_result, verdict.stage2_reason = spec.stage2
            verdict.stage2_tokens_used = 380 + 7 * i
            verdict.stage2_profile_version_id = profile_version_id
        if spec.key_skills is not None:
            verdict.stage2_key_skills = spec.key_skills
        if spec.stage3 is not None:
            # Validate against the real schema so a typo here fails loudly.
            result = Stage3Result.model_validate(spec.stage3).model_dump(mode="json")
            verdict.stage3_result = result
            verdict.stage3_triggered_at = now - timedelta(days=min(spec.days_ago, 2))
            verdict.stage3_tokens_used = 2400 + 31 * i
            verdict.stage3_profile_version_id = profile_version_id
        db.flush()

    for index, note in STARRED.items():
        star_posting(db, postings[index].id, note=note)

    for index, (status, applied_days_ago, follow_up_in, resume, notes) in APPLICATIONS.items():
        update_application_status(
            db,
            postings[index].id,
            status,
            resume_used=resume,
            date_applied=today - timedelta(days=applied_days_ago),
            follow_up_date=today + timedelta(days=follow_up_in),
            notes=notes,
        )

    roadmap = db.scalar(
        select(SkillRoadmap).where(SkillRoadmap.skill_or_language == "Apache Kafka")
    )
    if roadmap is None:
        kafka_evidence = [
            postings[i].id for i, p in enumerate(POSTINGS) if "Apache Kafka" in (p.key_skills or [])
        ]
        db.add(
            SkillRoadmap(
                skill_or_language="Apache Kafka",
                trigger_type="job_market_gap",
                is_new_territory=False,
                trend_evidence=kafka_evidence,
                status="in_progress",
                what_it_is="A distributed log used to move events between services in real time.",
                why_it_matters=(
                    "Streaming roles keep appearing in the postings and are the largest "
                    "recurring gap versus the candidate's batch-oriented experience."
                ),
                suggested_scope=(
                    "Rebuild one stage of the City Transit Lakehouse ingest as a Kafka consumer "
                    "with replay and idempotent writes."
                ),
                suggested_stack="Kafka, Python (confluent-kafka), Docker Compose, PostgreSQL",
                strengthens_project="City Transit Lakehouse",
                project_note="Adds a streaming ingest path next to the existing batch one.",
            )
        )
        db.flush()

    rescore_all(db)

    demo = Posting.external_id.like(f"{EXTERNAL_ID_PREFIX}%")
    demo_posting_ids = select(Posting.id).where(demo)
    in_demo = Verdict.posting_id.in_(demo_posting_ids)
    return {
        "profile_versions": _count(db, ProfileVersion, ProfileVersion.changelog == DEMO_CHANGELOG),
        "companies": _count(db, Company, Company.name.in_([c.name for c in COMPANIES])),
        "postings": _count(db, Posting, demo),
        "verdicts": _count(db, Verdict, in_demo),
        "stage2_results": _count(db, Verdict, in_demo, Verdict.stage2_result.is_not(None)),
        "stage2_key_skills": _count(db, Verdict, in_demo, Verdict.stage2_key_skills.is_not(None)),
        "stage3_results": _count(db, Verdict, in_demo, Verdict.stage3_result.is_not(None)),
        "starred": _count(db, Posting, demo, Posting.starred.is_(True)),
        "applications": _count(db, Application, Application.posting_id.in_(demo_posting_ids)),
        "roadmap_items": _count(db, SkillRoadmap, SkillRoadmap.skill_or_language == "Apache Kafka"),
    }


# --------------------------------------------------------------------------------------------
# CLI: separate database, hard guards.
# --------------------------------------------------------------------------------------------


def demo_database_url(main_url: str) -> URL:
    return make_url(main_url).set(database=DEMO_DATABASE)


def check_target_database(target: str | None, main: str | None) -> None:
    """Refuse anything that is not clearly a demo database or that is the configured main one."""
    if not target or not target.endswith("_demo"):
        raise SystemExit(f"refusing to seed {target!r}: the database name must end with '_demo'")
    if target == main:
        raise SystemExit(f"refusing to seed {target!r}: it is the configured main database")


def _admin_dsn(url: URL) -> str:
    return url.set(drivername="postgresql", database="postgres").render_as_string(
        hide_password=False
    )


def _recreate(url: URL, *, reset: bool) -> None:
    name = url.database
    with psycopg.connect(_admin_dsn(url), autocommit=True) as conn:
        if reset:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{name}"')


def _next_steps(url: URL) -> str:
    shown = url
    if url.password:
        shown = url.set(password="PASSWORD_PLACEHOLDER")
    dsn = shown.render_as_string(hide_password=False).replace("PASSWORD_PLACEHOLDER", "<password>")
    return f"""
Demo database ready: {url.database}

Next steps
  1. Generate your own API token:
       poetry run python scripts/generate_api_token.py
  2. Start the backend against the demo database (replace <password> and <token>):
       DATABASE_URL='{dsn}' \\
       API_TOKEN='<token>' \\
       poetry run uvicorn app.main:app --reload --reload-dir app
  3. Point the desktop app or extension at http://127.0.0.1:8000 with the same token.
  Re-run with --reset to wipe and rebuild the demo data. No Anthropic key is needed; do not use
  the Stage 2 / Stage 3 / batch buttons on demo data.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed fictional demo data into argus_demo.")
    parser.add_argument("--reset", action="store_true", help="drop and recreate the demo database")
    args = parser.parse_args()

    main_url = make_url(settings.database_url)
    url = demo_database_url(settings.database_url)
    check_target_database(url.database, main_url.database)

    _recreate(url, reset=args.reset)

    # Alembic's env.py reads settings.database_url, so point it at the demo database.
    settings.database_url = url.render_as_string(hide_password=False)
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")

    engine = create_engine(url)
    try:
        with Session(engine) as db:
            counts = seed(db)
            db.commit()
    finally:
        engine.dispose()

    print("Seeded:")
    for key, value in counts.items():
        print(f"  {key}: {value}")
    print(_next_steps(url))


if __name__ == "__main__":
    main()
