import math

import pytest
from sqlalchemy.orm import Session

from app.db import engine
from app.schemas.profile import Criteria, ProfileSnapshotData, Skills
from app.services import relevance
from app.services.postings import capture_posting
from app.services.profile import create_profile_version

DOCS = [
    "python machine learning pytorch deep learning models",  # doc_ml
    "java backend spring boot microservices rest api",  # doc_java
    "python data engineering etl pipelines airflow spark",  # doc_data
    "sales quota outreach crm pipeline marketing",  # doc_sales -- fully irrelevant distractor,
    # also keeps "python" from appearing in exactly 2-of-3 docs (which would
    # zero out its idf entirely: log(n/(1+df)) = log(3/3) = 0)
]
QUERY = "python machine learning pytorch"


def test_tokenize_lowercases_and_strips_punctuation():
    assert relevance.tokenize("Python, ML!") == ["python", "ml"]


def test_term_frequencies_normalizes_by_length():
    tf = relevance.term_frequencies(["a", "a", "b"])
    assert tf == {"a": 2 / 3, "b": 1 / 3}


def test_term_frequencies_empty_tokens_returns_empty():
    assert relevance.term_frequencies([]) == {}


def test_idf_gives_common_terms_lower_weight_than_rare_terms():
    df = relevance.document_frequencies([["python", "java"], ["python", "go"], ["python", "rust"]])
    idf = relevance.inverse_document_frequencies(df, n_docs=3)
    assert idf["python"] < idf["java"]  # python appears in all 3 docs, java in only 1


def test_tfidf_vector_drops_terms_absent_from_idf_vocabulary():
    tf = {"python": 0.5, "unseen_term": 0.5}
    idf = {"python": 1.2}
    vec = relevance.tfidf_vector(tf, idf)
    assert vec == {"python": 0.6}


def test_cosine_similarity_identical_vectors_is_one():
    vec = {"a": 1.0, "b": 2.0}
    assert math.isclose(relevance.cosine_similarity(vec, vec), 1.0)


def test_cosine_similarity_disjoint_vectors_is_zero():
    assert relevance.cosine_similarity({"a": 1.0}, {"b": 1.0}) == 0.0


def test_cosine_similarity_empty_vector_is_zero():
    assert relevance.cosine_similarity({}, {"a": 1.0}) == 0.0


def test_rank_by_relevance_orders_most_relevant_first():
    items = [
        ("doc_ml", DOCS[0]),
        ("doc_java", DOCS[1]),
        ("doc_data", DOCS[2]),
        ("doc_sales", DOCS[3]),
    ]
    ranked = relevance.rank_by_relevance(items, QUERY)
    # doc_ml matches all 4 query terms; doc_data shares only "python" (a
    # weaker signal once idf downweights it); doc_java/doc_sales share
    # nothing with the query and are tied at 0.0 similarity -- their
    # relative order isn't the point of this test, so only the top 2 (the
    # actual ranking distinction) are asserted.
    assert ranked[:2] == ["doc_ml", "doc_data"]


def test_rank_by_relevance_empty_items_returns_empty():
    assert relevance.rank_by_relevance([], QUERY) == []


def test_profile_query_text_includes_positive_signals_excludes_skip_patterns():
    profile = ProfileSnapshotData(
        experience=[],
        projects=[],
        skills=Skills(core=["Python"], developing=["Go"], coursework=["Java"]),
        criteria=Criteria(skip_patterns=["DevOps"], target_signals=["ML/AI engineering"]),
    )
    text = relevance.profile_query_text(profile)
    assert "Python" in text
    assert "Go" in text
    assert "ML/AI engineering" in text
    assert "DevOps" not in text
    assert "Java" not in text  # coursework isn't included either, only core/developing


def test_top_matched_terms_only_shared_terms_ranked_by_product():
    doc_vec = {"python": 0.8, "kubernetes": 0.5, "unrelated": 0.9}
    query_vec = {"python": 0.6, "kubernetes": 0.1}
    result = relevance.top_matched_terms(doc_vec, query_vec, n=5)
    assert set(result) == {"python", "kubernetes"}  # "unrelated" not in query_vec, excluded
    assert result[0] == "python"  # 0.8*0.6=0.48 > 0.5*0.1=0.05


def test_top_matched_terms_respects_n():
    doc_vec = {f"term{i}": float(i) for i in range(20)}
    query_vec = {f"term{i}": 1.0 for i in range(20)}
    assert len(relevance.top_matched_terms(doc_vec, query_vec, n=8)) == 8


def test_top_matched_terms_empty_when_no_overlap():
    assert relevance.top_matched_terms({"a": 1.0}, {"b": 1.0}) == []


# -- rescore_all (needs a real DB) --

EMPTY_SNAPSHOT = {
    "experience": [],
    "projects": [],
    "skills": {"core": [], "developing": [], "coursework": []},
    "credentials": {"education": [], "oss_contributions": [], "publications": []},
    "criteria": {
        "skip_patterns": [],
        "target_signals": [],
        "location_prefs": {},
        "seniority_floor": None,
        "seniority_ceiling": None,
    },
}


@pytest.fixture
def db():
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    create_profile_version(session, snapshot=EMPTY_SNAPSHOT)
    relevance._similarity_index_cache.clear()  # hygiene: no stale cross-test entries
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def test_rescore_all_sets_relevance_score_on_every_posting(db):
    posting, _ = capture_posting(
        db,
        company_name="RescoreCo1",
        title="ML Engineer",
        raw_jd_text="Build large-scale machine learning systems in Python.",
        url="https://rescoreco1.example.com/1",
    )
    result = relevance.rescore_all(db)
    assert result["rescored"] >= 1
    db.refresh(posting)
    assert posting.relevance_score is not None


def test_rescore_all_sets_relevance_profile_version_id(db):
    from app.services.profile import get_current_profile

    posting, _ = capture_posting(
        db,
        company_name="RescoreCo2",
        title="ML Engineer",
        raw_jd_text="text",
        url="https://rescoreco2.example.com/1",
    )
    relevance.rescore_all(db)
    db.refresh(posting)
    assert posting.relevance_profile_version_id == get_current_profile(db).id


def test_rescore_all_scores_higher_for_more_relevant_posting(db):
    create_profile_version(
        db,
        snapshot={
            **EMPTY_SNAPSHOT,
            "skills": {
                "core": ["kubernetes", "distributed", "systems"],
                "developing": [],
                "coursework": [],
            },
            "criteria": {
                **EMPTY_SNAPSHOT["criteria"],
                "target_signals": ["kubernetes", "distributed systems"],
            },
        },
    )
    relevant, _ = capture_posting(
        db,
        company_name="RescoreRelCo",
        title="ML Engineer",
        raw_jd_text="About the Role Build distributed systems on kubernetes at scale.",
        url="https://rescorerelco.example.com/1",
    )
    irrelevant, _ = capture_posting(
        db,
        company_name="RescoreIrrelCo",
        title="ML Engineer",
        raw_jd_text="About the Role Sales representative, enterprise accounts, quota driven.",
        url="https://rescoreirrelco.example.com/1",
    )
    relevance.rescore_all(db)
    db.refresh(relevant)
    db.refresh(irrelevant)
    assert relevant.relevance_score > irrelevant.relevance_score


def test_rescore_all_uses_normalized_jd_text_not_raw_jd_text(db):
    posting, _ = capture_posting(
        db,
        company_name="RescoreNormCo",
        title="ML Engineer",
        raw_jd_text="About the Role Generic boilerplate paragraph with no target skill words.",
        url="https://rescorenormco.example.com/1",
    )
    # Simulate normalized text diverging from raw (e.g. an older row whose
    # normalized_jd_text was recomputed from richer sections than what's in
    # raw_jd_text's summary alone) -- rescore_all must read the normalized
    # column, not re-derive from raw_jd_text itself.
    posting.normalized_jd_text = "kubernetes distributed systems kubernetes kubernetes"
    db.flush()
    create_profile_version(
        db,
        snapshot={
            **EMPTY_SNAPSHOT,
            "criteria": {**EMPTY_SNAPSHOT["criteria"], "target_signals": ["kubernetes"]},
        },
    )

    relevance.rescore_all(db)
    db.refresh(posting)
    assert posting.relevance_score > 0
    assert "kubernetes" in posting.relevance_matched_terms


def test_rescore_all_zero_score_when_normalized_text_is_empty(db):
    posting, _ = capture_posting(
        db,
        company_name="RescoreEmptyCo",
        title="ML Engineer",
        raw_jd_text="x",
        url="https://rescoreemptyco.example.com/1",
    )
    posting.normalized_jd_text = None
    db.flush()
    relevance.rescore_all(db)
    db.refresh(posting)
    assert posting.relevance_score == 0.0


# -- find_similar_postings (2.5) --


def _posting_with_text(db, name, url, *, title="ML Engineer", normalized_jd_text=""):
    posting, _ = capture_posting(
        db,
        company_name=name,
        title=title,
        raw_jd_text="x",
        url=url,
    )
    posting.normalized_jd_text = normalized_jd_text
    db.flush()
    return posting


def test_find_similar_postings_returns_none_for_unknown_id(db):
    assert relevance.find_similar_postings(db, 999999) is None


def test_find_similar_postings_orders_by_similarity(db):
    target = _posting_with_text(
        db,
        "SimTargetCo",
        "https://simtarget.example.com/1",
        normalized_jd_text="kubernetes distributed systems python",
    )
    close = _posting_with_text(
        db,
        "SimCloseCo",
        "https://simclose.example.com/1",
        normalized_jd_text="kubernetes distributed systems golang",
    )
    far = _posting_with_text(
        db,
        "SimFarCo",
        "https://simfar.example.com/1",
        normalized_jd_text="sales quota outreach crm marketing",
    )

    result = relevance.find_similar_postings(db, target.id, limit=5000)
    ids = [p.id for p, _sim, _flag in result]
    assert ids.index(close.id) < ids.index(far.id)


def test_find_similar_postings_excludes_the_target_itself(db):
    target = _posting_with_text(db, "SimSelfCo", "https://simself.example.com/1")
    result = relevance.find_similar_postings(db, target.id, limit=1000)
    assert all(p.id != target.id for p, _sim, _flag in result)


def test_find_similar_postings_respects_limit(db):
    target = _posting_with_text(db, "SimLimitCo", "https://simlimit.example.com/1")
    for i in range(5):
        _posting_with_text(db, f"SimLimitOtherCo{i}", f"https://simlimitother{i}.example.com/1")
    result = relevance.find_similar_postings(db, target.id, limit=3)
    assert len(result) == 3


def test_find_similar_postings_possibly_same_role_requires_same_company_and_threshold(db):
    from app.services.companies import get_or_create_company
    from app.services.postings import upsert_posting, write_stage1_verdict

    company = get_or_create_company(db, "SimSameCoDupTest")
    identical_text = "kubernetes kubernetes kubernetes distributed systems platform engineering"
    target, *_ = upsert_posting(
        db,
        company,
        title="Platform Engineer",
        url="https://simsamedup1.example.com/1",
        external_id=None,
        raw_jd_text="x",
        posted_date=None,
        source="greenhouse_api",
    )
    write_stage1_verdict(db, target)
    target.normalized_jd_text = identical_text
    same_company_dup, *_ = upsert_posting(
        db,
        company,
        title="Platform Engineer",
        url="https://simsamedup2.example.com/1",
        external_id=None,
        raw_jd_text="x",
        posted_date=None,
        source="greenhouse_api",
    )
    write_stage1_verdict(db, same_company_dup)
    same_company_dup.normalized_jd_text = identical_text
    db.flush()

    # A different company with the SAME title + text -- similarity is
    # just as high, but possibly_same_role must be False: different
    # companies can't be "the same role on another team/req" at one
    # company.
    other_company_same_text = _posting_with_text(
        db,
        "SimOtherCoDupTest",
        "https://simotherdup.example.com/1",
        title="Platform Engineer",
        normalized_jd_text=identical_text,
    )

    # limit generous enough that both comparison postings are guaranteed
    # to appear even against the real ~2,744-posting dev corpus -- their
    # near-1.0 similarity to target should rank them at the very top
    # regardless, but this removes any doubt.
    result = relevance.find_similar_postings(db, target.id, limit=5000)
    by_id = {p.id: (sim, flag) for p, sim, flag in result}

    sim_same, flag_same = by_id[same_company_dup.id]
    assert sim_same >= relevance.SIMILAR_ROLE_SAME_COMPANY_THRESHOLD
    assert flag_same is True

    sim_other, flag_other = by_id[other_company_same_text.id]
    assert sim_other >= relevance.SIMILAR_ROLE_SAME_COMPANY_THRESHOLD  # same similarity
    assert flag_other is False  # different company -- never flagged


def test_find_similar_postings_possibly_same_role_false_below_threshold(db):
    target = _posting_with_text(
        db,
        "SimBelowThreshCo",
        "https://simbelowthresh1.example.com/1",
        normalized_jd_text="python backend engineering apis",
    )
    from app.services.companies import get_or_create_company
    from app.services.postings import upsert_posting, write_stage1_verdict

    company = get_or_create_company(db, "SimBelowThreshCo")  # same company as target
    somewhat_related, *_ = upsert_posting(
        db,
        company,
        title="Data Engineer",
        url="https://simbelowthresh2.example.com/1",
        external_id=None,
        raw_jd_text="x",
        posted_date=None,
        source="greenhouse_api",
    )
    write_stage1_verdict(db, somewhat_related)
    somewhat_related.normalized_jd_text = "python data pipelines etl"  # related, not near-duplicate
    # TF-IDF weights only mean something over a real corpus; a two-document one makes every
    # shared term look identical.
    for i, text in enumerate(
        [
            "kubernetes terraform cloud networking",
            "sales quota enterprise accounts pipeline",
            "mechanical cad tolerance manufacturing",
            "legal contracts compliance counsel",
            "frontend react typescript design systems",
            "financial modeling accounting audit",
        ]
    ):
        filler = _posting_with_text(
            db, f"SimFillerCo{i}", f"https://simfiller{i}.example.com/1", normalized_jd_text=text
        )
        assert filler.id
    db.flush()

    result = relevance.find_similar_postings(db, target.id, limit=5000)
    sim, flag = next((s, f) for p, s, f in result if p.id == somewhat_related.id)
    assert sim < relevance.SIMILAR_ROLE_SAME_COMPANY_THRESHOLD
    assert flag is False


def test_find_similar_postings_survives_across_separate_sessions():
    """Regression test: the similarity cache once held ORM Posting objects from the session
    that built it, so a second, separate session hitting the cache raised
    DetachedInstanceError on any lazy-loaded attribute. The cache now stores plain ids/vectors
    and re-fetches fresh ORM objects from whichever session asks. This test uses independent
    connections and committed rows (not the shared rollback `db` fixture) to reproduce "a
    later, different request", and removes what it commits."""
    from sqlalchemy import delete, select
    from sqlalchemy.orm import Session as ORMSession

    from app.db import engine as real_engine
    from app.models.company import Company
    from app.models.posting import Posting
    from app.models.verdict import Verdict

    relevance._similarity_index_cache.clear()
    created_ids: list[int] = []
    try:
        with real_engine.begin() as conn1:  # commits on exit
            session1 = ORMSession(bind=conn1)
            create_profile_version(session1, snapshot=EMPTY_SNAPSHOT)
            for i in range(2):
                p = _posting_with_text(
                    session1, f"SimCrossSessCo{i}", f"https://simcrosssess{i}.example.com/1"
                )
                created_ids.append(p.id)
            session1.flush()
            # First call: populates the cache from session1.
            assert relevance.find_similar_postings(session1, created_ids[0], limit=5) is not None
            session1.close()

        # Second call: a completely separate session. If the cache held session1's ORM
        # objects, touching a lazy-loaded attribute here would raise DetachedInstanceError.
        with real_engine.connect() as conn2:
            session2 = ORMSession(bind=conn2)
            second = relevance.find_similar_postings(session2, created_ids[0], limit=3)
            assert second
            for posting, _sim, _flag in second:
                assert posting.company is not None
            session2.close()
    finally:
        relevance._similarity_index_cache.clear()
        with real_engine.begin() as conn3:
            cleanup = ORMSession(bind=conn3)
            company_ids = list(
                cleanup.scalars(select(Posting.company_id).where(Posting.id.in_(created_ids)))
            )
            cleanup.execute(delete(Verdict).where(Verdict.posting_id.in_(created_ids)))
            cleanup.execute(delete(Posting).where(Posting.id.in_(created_ids)))
            cleanup.execute(delete(Company).where(Company.id.in_(company_ids)))
            cleanup.close()


# -- BM25 (bm25s) --


def test_bm25_scores_term_frequency_saturates():
    texts = ["python " * 3 + "filler " * 7, "python " * 30 + "filler " * 70, "java " * 10]
    s = relevance.bm25_scores(texts, "python")
    assert s[2] == 0.0
    assert s[0] > 0.0
    # 10x the occurrences at the same density must not score 10x.
    assert s[1] < 2 * s[0]


def test_bm25_scores_rare_term_outweighs_common_term():
    texts = ["common rare", "common", "common", "common", "other"]
    s = relevance.bm25_scores(texts, "common rare")
    assert s[0] > s[1]


def test_bm25_scores_empty_corpus_and_empty_query():
    assert relevance.bm25_scores([], "a query") == []
    # An empty profile has an empty query; it matches nothing rather than raising.
    assert relevance.bm25_scores(["alpha beta", "gamma"], "") == [0.0, 0.0]
    assert relevance.bm25_scores(["alpha beta", "gamma"], "the of") == [0.0, 0.0]  # stopwords only
