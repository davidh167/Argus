"""Mock tests for the embedding benchmark harness (research/native-engine/embeddings): stubbed
embedder, no Ollama, no model, no real database rows."""

import importlib.util
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app import native
from app.config import settings
from app.db import engine
from app.services import embeddings
from app.services.postings import capture_posting
from app.services.profile import create_profile_version
from tests.test_roadmap import EMPTY_SNAPSHOT

BENCH_DIR = Path(__file__).resolve().parents[1] / "research/native-engine/embeddings"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, BENCH_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_phrases_are_distinct_and_deterministic():
    bench = _load("bench_embeddings")
    phrases = bench.synthetic_phrases(500)
    assert len(set(phrases)) == 500
    assert phrases == bench.synthetic_phrases(500)


def test_a_missing_phrase_file_says_how_to_export_it(tmp_path):
    bench = _load("bench_embeddings")
    with pytest.raises(SystemExit, match="export_phrases.py"):
        bench.read_phrases(tmp_path / "nope.txt")


@pytest.mark.skipif(not native.DEFAULT_LIB.exists(), reason="needs the built libargus")
def test_harness_runs_every_scenario_with_a_stub_embedder(tmp_path, monkeypatch):
    bench = _load("bench_embeddings")
    for obj, attr in [
        (settings, "embeddings_backend"),
        (settings, "argus_index_path"),
        (settings, "argus_lib_path"),
        (embeddings, "_embed_batch"),
        (embeddings, "_open_persistent"),
        (embeddings, "_persistent_cache"),
    ]:
        monkeypatch.setattr(obj, attr, getattr(obj, attr))  # so the harness's changes are undone
    monkeypatch.setattr(settings, "argus_lib_path", None)
    monkeypatch.setattr(embeddings, "_embedding_cache", {})
    monkeypatch.setattr(embeddings, "_persistent", None)

    calls = []
    real_stub = bench.stub_embed
    monkeypatch.setattr(bench, "stub_embed", lambda b: calls.append(len(b)) or real_stub(b))
    phrases = bench.synthetic_phrases(300)
    harness = bench.Harness(tmp_path, stub=True)
    results = bench.run(phrases, repeats=2, include_ollama=False, harness=harness)

    assert [r["scenario"] for r in results] == [
        "native cold, no persistence",
        "native cold, persisting",
        "native restart, persistence",
        "native restart, no persistence",
        "in-process warm",
    ]
    assert all(len(r["times"]) == 2 for r in results)
    # 2 repeats x (cold no-persist + cold persist + restart no-persist) each embed all 300 phrases;
    # restart-with-persistence and warm embed nothing.
    assert sum(calls) == 3 * 2 * 300
    assert "| native restart, persistence |" in bench.report(results, 300, 2, stub=True)


def test_export_collects_distinct_normalized_phrases_from_latest_verdicts():
    export = _load("export_phrases")
    connection = engine.connect()
    transaction = connection.begin()
    db = Session(bind=connection)
    create_profile_version(db, snapshot=EMPTY_SNAPSHOT)
    try:
        for i, skills in enumerate([["PyTorch", "PCB layout"], ["pytorch", "RF-design"], []]):
            posting, _ = capture_posting(
                db,
                company_name=f"ExportCo{i}",
                title="ML Engineer",
                raw_jd_text="x",
                url=f"https://exportco{i}.example.com/1",
            )
            posting.verdicts[0].stage2_key_skills = skills
        db.flush()
        assert export.distinct_phrases(db) == ["PCB layout", "PyTorch", "RF-design"]
    finally:
        db.close()
        transaction.rollback()
        connection.close()
