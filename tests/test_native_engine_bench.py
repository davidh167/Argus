"""Pieces of the Phase 0 benchmark harness (research/native-engine): corpus generation, the
throwaway-database guard, report formatting, the non-ASCII scan, and a tiny end-to-end run of the
timed cases. Small N throughout; these test the harness, not the timings."""

import json

import numpy as np
import pytest
from sqlalchemy.orm import Session

import bench_corpus
import bench_db
import bench_report
import bench_scan
import bench_stub
from app.db import engine
from app.services import relevance
from bench_cases import run_cases


@pytest.fixture
def session_factory():
    """Sessions that share one outer transaction, rolled back at the end, so the corpus the tests
    load never reaches the test database for other tests."""
    conn = engine.connect()
    outer = conn.begin()

    def factory() -> Session:
        return Session(bind=conn, join_transaction_mode="create_savepoint")

    yield factory
    outer.rollback()
    conn.close()


# --- corpus ---------------------------------------------------------------------------------


def test_synthetic_corpus_is_deterministic_per_seed():
    a = bench_corpus.generate_synthetic(60, seed=7)
    b = bench_corpus.generate_synthetic(60, seed=7)
    c = bench_corpus.generate_synthetic(60, seed=8)
    assert a.postings == b.postings
    assert a.skills_core == b.skills_core
    assert a.postings != c.postings


def test_synthetic_corpus_shape():
    corpus = bench_corpus.generate_synthetic(200, seed=1)
    assert [p.id for p in corpus.postings] == list(range(1, 201))
    assert all(p.key_skills and p.text and p.title for p in corpus.postings)
    assert any(p.positive for p in corpus.postings)
    with_non_ascii = sum(1 for p in corpus.postings if not p.text.isascii())
    assert 0 < with_non_ascii < 60  # about 6% were asked for


def test_synthetic_corpus_without_non_ascii_fraction_is_ascii():
    corpus = bench_corpus.generate_synthetic(100, seed=1, nonascii_fraction=0.0)
    # Titles can still get a non-ASCII suffix; the JD text cannot.
    assert all(p.text.isascii() for p in corpus.postings)


def test_load_snapshot_reads_export_format(tmp_path):
    rows = [
        {
            "id": 10 + i,
            "title": f"Role {i}",
            "company": "Acme",
            "source": "greenhouse_api",
            "closed": False,
            "text": f"python café {i}",
            "positive": i == 0,
            "stage1": "analyze",
            "stage2": None,
            "relevance_score": None,
        }
        for i in range(5)
    ]
    (tmp_path / "snapshot.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "meta.json").write_text(
        json.dumps({"exported_at": "2026-10-29T00:00:00+00:00", "profile_query": "python search"})
    )
    corpus = bench_corpus.load_snapshot(tmp_path)
    assert [p.id for p in corpus.postings] == [10, 11, 12, 13, 14]
    assert corpus.postings[0].positive and not corpus.postings[1].positive
    assert corpus.postings[0].text == "python café 0"
    assert corpus.target_signals == ["python search"]
    assert all(p.key_skills for p in corpus.postings)
    assert bench_corpus.load_snapshot(tmp_path).postings == corpus.postings


# --- throwaway database guard ---------------------------------------------------------------


def test_bench_database_name_defaults_to_suffixed_name():
    assert bench_db.bench_database_name("job_search") == "job_search_bench"
    assert bench_db.bench_database_name("job_search", "other") == "other"


def test_bench_database_name_refuses_the_configured_database():
    with pytest.raises(ValueError, match="configured database"):
        bench_db.bench_database_name("job_search", "job_search")


def test_throwaway_database_refuses_once_the_engine_is_bound():
    # app.db is imported (this module imports it), so its engine already points at some URL.
    with pytest.raises(RuntimeError, match="already imported"):
        with bench_db.throwaway_database():
            pass


# --- report ---------------------------------------------------------------------------------


def _timing(name: str, samples: list[float]) -> bench_report.Timing:
    return bench_report.Timing(name, "what", samples, warmup=3)


def test_percentile_and_formatting():
    samples = [float(i) for i in range(1, 31)]
    assert bench_report.percentile(samples, 50) == 15.5
    assert bench_report.percentile(samples, 95) == pytest.approx(28.55)
    assert bench_report.fmt_ms(0.1234) == "0.12"
    assert bench_report.fmt_ms(12.34) == "12.3"
    assert bench_report.fmt_ms(123.4) == "123"
    assert bench_report.fmt_ms(12345.6) == "12,346"


def test_timings_table_rows():
    table = bench_report.timings_table([_timing("case a", [float(i) for i in range(1, 31)])])
    assert "| case a | 15.5 | 28.5 | 1.00 | 30.0 | 30 | 3 |" in table


def _cold_warm(cold_ms: float, warm_ms: float) -> list[bench_report.Timing]:
    return [
        _timing(bench_report.COLD_CASE, [cold_ms] * 30),
        _timing(bench_report.WARM_CASE, [warm_ms] * 30),
    ]


def test_kill_criterion_satisfied_when_both_limits_hold():
    text = bench_report.kill_criterion(_cold_warm(cold_ms=900.0, warm_ms=49.0))
    assert "satisfied: yes" in text


@pytest.mark.parametrize("cold_ms,warm_ms", [(1000.0, 10.0), (500.0, 50.0), (2000.0, 80.0)])
def test_kill_criterion_not_satisfied_at_or_over_either_limit(cold_ms, warm_ms):
    text = bench_report.kill_criterion(_cold_warm(cold_ms, warm_ms))
    assert "satisfied: no" in text


def test_render_report_contains_environment_and_sections():
    report = bench_report.render_report({"machine": "test box", "python": "3.12"}, ["## A", "## B"])
    assert report.startswith("# Python baseline for the native engine (Phase 0)")
    assert "- machine: test box" in report
    assert report.index("## A") < report.index("## B")


# --- non-ASCII scan -------------------------------------------------------------------------


def test_scan_tokenizers_match_relevance_for_unicode():
    text = "Café résumé 数据 engineer, C++ and python_3!"
    assert bench_scan.tokenize_unicode(text) == relevance.tokenize(text)
    assert bench_scan.tokenize_ascii(text) == [
        "caf",
        "r",
        "sum",
        "engineer",
        "c",
        "and",
        "python_3",
    ]
    assert bench_scan.similarity_text("T", "body") == relevance._similarity_text("T", "body")


def test_scan_counts_non_ascii_tokens():
    docs = [
        ("Engineer", "python café backend"),
        ("Engineer", "python backend services"),
        ("Designer", "figma wireframes 数据"),
        ("Analyst", "sql dashboards reports"),
    ]
    result = bench_scan.scan(docs, progress=lambda _msg: None)
    assert result.n_postings == 4
    assert result.n_postings_non_ascii == 2
    assert result.n_tokens_non_ascii == 2
    assert result.n_distinct_non_ascii_tokens == 2


def test_scan_reports_no_change_for_ascii_corpus():
    corpus = bench_corpus.generate_synthetic(40, seed=3, nonascii_fraction=0.0)
    result = bench_scan.scan(
        [("Engineer", p.text) for p in corpus.postings], progress=lambda _m: None
    )
    assert result.n_postings_non_ascii == 0
    assert result.n_top10_order_changed == 0
    assert result.n_top10_set_changed == 0
    assert result.n_top1_changed == 0


def test_scan_top_k_matches_relevance_find_similar(session_factory):
    corpus = bench_corpus.generate_synthetic(60, seed=5)
    with session_factory() as db:
        bench_db.load_corpus(db, corpus)
        db.commit()
    relevance._similarity_index_cache.clear()
    token_lists = [
        bench_scan.tokenize_unicode(bench_scan.similarity_text(p.title, p.text))
        for p in corpus.postings
    ]
    neighbors = bench_scan.top_k_neighbors(token_lists, k=10)
    for index in (0, 17, 42):
        with session_factory() as db:
            expected = relevance.find_similar_postings(db, corpus.postings[index].id, limit=10)
        assert expected is not None
        assert [corpus.postings[i].id for i in neighbors[index]] == [p.id for p, _s, _f in expected]
    relevance._similarity_index_cache.clear()


# --- stub embedder --------------------------------------------------------------------------


def test_stub_embedder_is_deterministic_unit_norm_and_word_sensitive():
    a = bench_stub.stub_embed_texts(["distributed systems design", "systems design", "pcb layout"])
    b = bench_stub.stub_embed_texts(["distributed systems design", "systems design", "pcb layout"])
    assert a.shape == (3, bench_stub.DIM)
    assert np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert a[0] @ a[1] > a[0] @ a[2]


# --- cases ----------------------------------------------------------------------------------


def test_run_cases_end_to_end_on_a_tiny_corpus(session_factory):
    corpus = bench_corpus.generate_synthetic(40, seed=2)
    with session_factory() as db:
        bench_db.load_corpus(db, corpus)
        db.commit()
    relevance._similarity_index_cache.clear()

    results = run_cases(session_factory, runs=2, warmup=1)

    names = [t.name for t in results.timings]
    assert names[:5] == [
        bench_report.COLD_CASE,
        bench_report.WARM_CASE,
        "rescore_all",
        "bm25_scores",
        "market_demand (stub embedder)",
    ]
    assert len(names) == 9  # five cases + four find_similar components
    assert all(len(t.samples_ms) == 2 and t.warmup == 1 for t in results.timings)
    assert all(s > 0 for t in results.timings for s in t.samples_ms)
    assert results.facts["rescore_all rescored"] == "40"
    assert int(results.facts["market_demand clusters"]) > 0
    relevance._similarity_index_cache.clear()
