"""Corpora for the Phase 0 baseline: a seeded synthetic generator and the ranking-eval snapshot loader.

Neither touches a database. `bench_db.load_corpus` writes a corpus into the throwaway benchmark DB.

The synthetic corpus is shaped like the real one where it matters for timing, not for meaning:
Zipf-distributed word frequencies over a vocabulary with a long rare tail, posting lengths in the
hundreds of words, topic-clustered vocabulary (so similarity scores are not uniform noise), some
near-duplicate postings inside a company, a few percent non-ASCII text, and key-skill phrase counts
close to the ratio seen on real data (about 16k phrase instances over 7.5k distinct for ~3k postings).
"""

import json
import random
from dataclasses import dataclass, field
from itertools import accumulate
from pathlib import Path

DEFAULT_SEED = 20261101

# Domain-neutral topic vocabularies: each posting draws mostly from one topic's words.
TOPICS: dict[str, list[str]] = {
    "backend": (
        "python java golang rust postgres redis kafka microservices api rest grpc docker kubernetes "
        "latency throughput scalability concurrency caching sharding queue schema migration "
        "observability tracing terraform"
    ).split(),
    "frontend": (
        "react typescript javascript css html accessibility responsive webpack vite component "
        "state routing animation figma design-system browser rendering performance mobile ios "
        "android swift kotlin"
    ).split(),
    "data": (
        "pytorch tensorflow embeddings retrieval ranking classification regression pipeline spark "
        "airflow warehouse feature experiment metrics statistics sql dashboard annotation labeling "
        "inference training evaluation"
    ).split(),
    "hardware": (
        "firmware embedded rtos pcb schematic fpga verilog microcontroller sensor bluetooth "
        "oscilloscope soldering emc thermal enclosure prototype manufacturing bringup debugging "
        "analog power"
    ).split(),
    "sales": (
        "quota pipeline prospecting crm negotiation enterprise accounts renewal territory outreach "
        "demo closing forecast revenue upsell partnership discovery objection channel onboarding "
        "stakeholders"
    ).split(),
    "marketing": (
        "campaign brand seo content audience funnel conversion newsletter social analytics "
        "positioning launch webinar copywriting lifecycle attribution segmentation influencer "
        "creative storytelling events"
    ).split(),
    "finance": (
        "accounting ledger reconciliation audit forecasting budget payroll tax compliance gaap "
        "treasury invoice variance reporting controls procurement cashflow liquidity risk "
        "valuation modeling"
    ).split(),
    "healthcare": (
        "clinical patient nursing triage charting hipaa scheduling billing coding pharmacy "
        "diagnosis outcomes care-plan telehealth intake referral therapy compliance wellness "
        "rounds protocols"
    ).split(),
    "design": (
        "wireframe prototype usability research persona journey typography layout interaction "
        "visual branding illustration motion critique workshop accessibility iteration moodboard "
        "grid palette handoff"
    ).split(),
    "operations": (
        "logistics warehouse inventory fulfillment shipping vendor sourcing forecasting lean "
        "kaizen safety scheduling routing fleet quality sop throughput procurement receiving "
        "dispatch planning"
    ).split(),
    "support": (
        "tickets escalation troubleshooting zendesk sla onboarding knowledge-base chat triage "
        "resolution empathy documentation macros csat queue refunds incident feedback training "
        "communication"
    ).split(),
    "security": (
        "vulnerability pentest threat siem firewall encryption iam incident forensics malware "
        "compliance soc2 detection response hardening phishing audit keys zero-trust patching "
        "monitoring"
    ).split(),
}

TITLE_NOUNS: dict[str, list[str]] = {
    "backend": ["Backend Engineer", "Platform Engineer", "Software Engineer"],
    "frontend": ["Frontend Engineer", "Mobile Engineer", "UI Engineer"],
    "data": ["Machine Learning Engineer", "Data Scientist", "Data Engineer"],
    "hardware": ["Firmware Engineer", "Hardware Engineer", "Electrical Engineer"],
    "sales": ["Account Executive", "Sales Development Representative", "Sales Manager"],
    "marketing": ["Marketing Manager", "Content Strategist", "Growth Marketer"],
    "finance": ["Financial Analyst", "Accountant", "Controller"],
    "healthcare": ["Registered Nurse", "Clinical Coordinator", "Patient Advocate"],
    "design": ["Product Designer", "UX Researcher", "Visual Designer"],
    "operations": ["Operations Manager", "Logistics Coordinator", "Supply Chain Analyst"],
    "support": ["Support Specialist", "Customer Success Manager", "Support Engineer"],
    "security": ["Security Engineer", "Security Analyst", "Incident Responder"],
}
SENIORITY = ["", "", "Senior", "Junior", "Staff", "Lead", "Principal"]

COMMON_WORDS = (
    "the and of to in for with a an is are will be you we our your this that on as at by or from "
    "it can have has experience team work build develop requirements ability strong skills "
    "knowledge years role responsibilities qualifications looking join help ensure support "
    "manage drive create collaborate communicate across products customers business projects "
    "tools process improve deliver quality use using including such other new high best "
    "environment opportunity position candidate preferred degree related field equivalent "
    "plus excellent understanding working closely cross-functional stakeholders priorities "
    "company mission growth impact fast-paced remote hybrid benefits salary location"
).split()

# Non-ASCII word characters: Latin diacritics, other Latin-script scripts, CJK, Cyrillic.
NON_ASCII_WORDS = (
    "café résumé naïve señor münchen zürich são paulo montréal québec córdoba düsseldorf "
    "ingénieur développeur entwicklerin überprüfung łódź gdańsk İstanbul 数据 工程师 エンジニア "
    "разработчик 개발자 مهندس"
).split()

_SYLLABLES = (
    "ba be bi bo bu ca ce ci co cu da de di do du fa fe fi fo fu ga ge gi go gu la le li lo lu "
    "ma me mi mo mu na ne ni no nu pa pe pi po pu ra re ri ro ru sa se si so su ta te ti to tu "
    "va ve vi vo vu xa xe xi za ze zi zo ar er or in on an el al ex"
).split()


@dataclass(frozen=True)
class CorpusPosting:
    id: int
    title: str
    company: str
    source: str
    text: str
    positive: bool
    key_skills: list[str]


@dataclass
class Corpus:
    name: str
    postings: list[CorpusPosting]
    target_signals: list[str]
    skills_core: list[str]
    skills_developing: list[str]
    project_stack: list[str]
    # Free-form facts about how the corpus was built, printed in the report.
    notes: dict[str, str] = field(default_factory=dict)


def _zipf_cum_weights(n: int) -> list[float]:
    return list(accumulate(1.0 / (rank + 1) for rank in range(n)))


def _tail_vocabulary(rng: random.Random, size: int) -> list[str]:
    """Pseudo-words standing in for the long tail of rare terms real text has (names, tools,
    one-off phrases). Deterministic from rng."""
    words: set[str] = set()
    while len(words) < size:
        words.add("".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(2, 4))))
    return sorted(words)


def _skill_pool(rng: random.Random, topics: list[str], size: int) -> dict[str, list[str]]:
    """topic -> distinct 2-3 word skill phrases built from that topic's words."""
    # Capped below the number of distinct ordered 2-3 word phrases the smallest topic can form.
    per_topic = min(max(size // len(topics), 20), 5000)
    pool: dict[str, list[str]] = {}
    for topic in topics:
        phrases: list[str] = []
        seen: set[str] = set()
        while len(phrases) < per_topic:
            words = rng.sample(TOPICS[topic], rng.choice([2, 2, 3]))
            phrase = " ".join(words)
            if phrase not in seen:
                seen.add(phrase)
                phrases.append(phrase)
        pool[topic] = phrases
    return pool


def _key_skills(rng: random.Random, pool: list[str], cum: list[float]) -> list[str]:
    skills = rng.choices(pool, cum_weights=cum, k=rng.randint(4, 7))
    # Real phrases come in mixed casing; _normalize_phrase collapses it before embedding.
    return [s.title() if rng.random() < 0.15 else s for s in dict.fromkeys(skills)]


def _profile_parts(
    rng: random.Random, topics: list[str], skill_pool: dict[str, list[str]]
) -> dict[str, list[str]]:
    target = topics[0]
    return {
        "skills_core": skill_pool[target][:15],
        "skills_developing": skill_pool[target][15:25],
        "project_stack": rng.sample(TOPICS[target], 6),
        "target_signals": [" ".join(rng.sample(TOPICS[target], 4)) for _ in range(3)],
    }


def generate_synthetic(n: int, seed: int = DEFAULT_SEED, nonascii_fraction: float = 0.06) -> Corpus:
    """n postings, fully determined by (n, seed, nonascii_fraction)."""
    rng = random.Random(seed)
    topics = list(TOPICS)
    tail = _tail_vocabulary(rng, 12_000)
    tail_cum = _zipf_cum_weights(len(tail))
    topic_cum = {t: _zipf_cum_weights(len(w)) for t, w in TOPICS.items()}
    common_cum = _zipf_cum_weights(len(COMMON_WORDS))
    skill_pool = _skill_pool(rng, topics, size=8 * n)
    skill_cum = {t: _zipf_cum_weights(len(p)) for t, p in skill_pool.items()}
    n_companies = max(n // 25, 5)

    postings: list[CorpusPosting] = []
    words_by_id: dict[int, list[str]] = {}
    meta: dict[int, tuple[str, int, str]] = {}  # id -> (topic, company index, title)
    for pid in range(1, n + 1):
        if postings and rng.random() < 0.10:
            # Near-duplicate of an earlier posting at the same company (a re-listed or sibling
            # requisition): same words with ~8% replaced.
            base = rng.choice(postings)
            topic, company_idx, title = meta[base.id]
            words = [
                w if rng.random() > 0.08 else rng.choices(tail, cum_weights=tail_cum)[0]
                for w in words_by_id[base.id]
            ]
        else:
            topic = rng.choice(topics)
            company_idx = rng.randrange(n_companies)
            seniority = rng.choice(SENIORITY)
            title = f"{seniority} {rng.choice(TITLE_NOUNS[topic])}".strip()
            length = max(80, int(rng.lognormvariate(6.0, 0.45)))  # median ~400 words
            n_topic, n_other, n_common = int(length * 0.40), int(length * 0.08), int(length * 0.30)
            other = rng.choice(topics)
            words = (
                rng.choices(TOPICS[topic], cum_weights=topic_cum[topic], k=n_topic)
                + rng.choices(TOPICS[other], cum_weights=topic_cum[other], k=n_other)
                + rng.choices(COMMON_WORDS, cum_weights=common_cum, k=n_common)
                + rng.choices(tail, cum_weights=tail_cum, k=length - n_topic - n_other - n_common)
            )
            rng.shuffle(words)
            if rng.random() < nonascii_fraction:
                for _ in range(rng.randint(1, 3)):
                    words.insert(rng.randrange(len(words)), rng.choice(NON_ASCII_WORDS))
        if rng.random() < 0.01:
            title = f"{title} ({rng.choice(NON_ASCII_WORDS)})"
        words_by_id[pid] = words
        meta[pid] = (topic, company_idx, title)
        text = _sentences(rng, words)
        positive = rng.random() < 0.02
        postings.append(
            CorpusPosting(
                id=pid,
                title=title,
                company=f"Company {company_idx:04d}",
                source="manual_capture" if positive else "greenhouse_api",
                text=text,
                positive=positive,
                key_skills=_key_skills(rng, skill_pool[topic], skill_cum[topic]),
            )
        )
    parts = _profile_parts(rng, topics, skill_pool)
    return Corpus(
        name=f"synthetic-{n}",
        postings=postings,
        notes={"seed": str(seed), "non-ASCII posting fraction": str(nonascii_fraction)},
        **parts,
    )


def _sentences(rng: random.Random, words: list[str]) -> str:
    """Join words into sentences of 8-20 words so the text has the punctuation real JDs have."""
    out: list[str] = []
    i = 0
    while i < len(words):
        k = rng.randint(8, 20)
        chunk = words[i : i + k]
        out.append(" ".join(chunk).capitalize() + ".")
        i += k
    return " ".join(out)


def load_snapshot(data_dir: Path, seed: int = DEFAULT_SEED) -> Corpus:
    """The snapshot written by research/ranking-eval/export_snapshot.py (snapshot.jsonl + meta.json).

    Real titles, JD text and interest labels. The snapshot carries no stage2_key_skills (only the
    verdict word), so market_demand needs them from somewhere: they are synthesized with the same
    generator as the synthetic corpus, and the profile's target_signals is the real profile query
    from meta.json. The report says so; market_demand on a snapshot run therefore measures the
    algorithm at the snapshot's N, not the real phrase distribution.
    """
    meta = json.loads((data_dir / "meta.json").read_text())
    rows = [json.loads(line) for line in (data_dir / "snapshot.jsonl").read_text().splitlines()]
    rng = random.Random(seed)
    topics = list(TOPICS)
    skill_pool = _skill_pool(rng, topics, size=8 * len(rows))
    skill_cum = {t: _zipf_cum_weights(len(p)) for t, p in skill_pool.items()}
    postings = [
        CorpusPosting(
            id=row["id"],
            title=row["title"],
            company=row["company"],
            source=row["source"],
            text=row["text"],
            positive=row["positive"],
            key_skills=_key_skills(rng, skill_pool[topic], skill_cum[topic]),
        )
        for row in rows
        for topic in [rng.choice(topics)]
    ]
    parts = _profile_parts(rng, topics, skill_pool)
    parts["target_signals"] = [meta["profile_query"]]
    return Corpus(
        name=f"snapshot-{len(postings)}",
        postings=postings,
        notes={
            "exported_at": meta["exported_at"],
            "key_skills": "synthesized (the snapshot has no stage2_key_skills)",
        },
        **parts,
    )
