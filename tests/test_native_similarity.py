"""Parity of the native similarity engine (libargus argus_sim_*) with app/services/relevance.py.

The native tokenizer must reproduce `re.findall(r"\\w+", text.lower())` and the scoring must reproduce
the Python tf-idf cosine ranking exactly. Three layers: every code point against Python's own `\\w` and
`lower()`, tricky strings, and whole corpora (every top-10 list identical, scores within 1e-9). All of it
needs the built library (native/README.md); CI builds the llama-free one.
"""

import random
import re
import unicodedata

import pytest

import bench_corpus
import bench_parity
from app import native
from app.native import NativeError
from app.services import relevance

needs_lib = pytest.mark.skipif(
    not native.lib_path().exists(), reason="needs the built libargus (native/README.md)"
)
pytestmark = needs_lib

SURROGATES = range(0xD800, 0xE000)
WORD = re.compile(r"\w")


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


# --- every code point ---------------------------------------------------------------------------


def test_every_code_point_agrees_with_python():
    """\\w membership and the full lowercase mapping, for U+0000..U+10FFFF except surrogates.

    Python 3.12 ships Unicode 15.0; ICU ships a newer one. Code points ASSIGNED in Python's version must
    agree exactly. Code points Python's database does not know (category Cn) may differ, because ICU
    already assigns them (new letters and their case mappings); that is Unicode version skew, and real
    postings do not contain characters Python 3.12 cannot classify. The test fails if the library's
    Unicode is OLDER than Python's (then assigned code points could differ for the wrong reason)."""
    icu_unicode, icu = native.unicode_versions()
    py_unicode = unicodedata.unidata_version
    assert _version(icu_unicode) >= _version(py_unicode), (
        f"ICU {icu} implements Unicode {icu_unicode}, older than Python's {py_unicode}"
    )

    word_skew: list[int] = []
    lower_skew: list[int] = []
    checked = 0
    for cp in range(0x110000):
        if cp in SURROGATES:
            continue
        char = chr(cp)
        is_word, lower = native.unicode_probe(cp)
        checked += 1
        if is_word != bool(WORD.fullmatch(char)):
            word_skew.append(cp)
        if lower != char.lower():
            lower_skew.append(cp)

    def unassigned_in_python(cps: list[int]) -> list[int]:
        return [cp for cp in cps if unicodedata.category(chr(cp)) == "Cn"]

    assigned_word = [cp for cp in word_skew if cp not in set(unassigned_in_python(word_skew))]
    assigned_lower = [cp for cp in lower_skew if cp not in set(unassigned_in_python(lower_skew))]
    print(
        f"\n{checked} code points checked; Python Unicode {py_unicode}, ICU {icu} (Unicode {icu_unicode}); "
        f"\\w disagreements {len(word_skew)} (assigned in Python: {len(assigned_word)}), "
        f"lower() disagreements {len(lower_skew)} (assigned in Python: {len(assigned_lower)}); "
        f"first skewed: {[f'U+{cp:04X}' for cp in (word_skew + lower_skew)[:5]]}"
    )
    assert assigned_word == [], [f"U+{cp:04X}" for cp in assigned_word[:20]]
    assert assigned_lower == [], [f"U+{cp:04X}" for cp in assigned_lower[:20]]
    if _version(icu_unicode) == _version(py_unicode):
        assert word_skew == [] and lower_skew == []


# --- tricky strings -----------------------------------------------------------------------------

TRICKY = {
    "dotted capital I": "\u0130stanbul \u0130ZM\u0130R",
    "dotless i": "\u0131s\u0131 I\u0131",
    "sharp s": "Stra\u00dfe STRASSE \u1e9e",
    "final sigma at word end": "\u039f\u0394\u039f\u03a3 \u039f\u0394\u039f\u03a3.",
    "sigma in the middle": "\u03a3\u03a3\u0391 \u0391\u03a3\u0391",
    "sigma after a combining mark": "\u0391\u0301\u03a3 \u0391\u0301\u03a3\u0391",
    "sigma alone": "\u03a3 \u03a3\u03a3 x\u03a3 \u03a3x",
    "cjk": "\u81ea\u7136\u8a00\u8a9e\u51e6\u7406 \u30a8\u30f3\u30b8\u30cb\u30a2 \ud55c\uad6d\uc5b4",
    "arabic": "\u0645\u0647\u0646\u062f\u0633 \u0627\u0644\u0628\u0631\u0645\u062c\u064a\u0627\u062a \u0661\u0662\u0663",
    "devanagari with combining marks": "\u0907\u0902\u091c\u0940\u0928\u093f\u092f\u0930 \u0915\u094d\u0937",
    "superscripts and fractions": "x\u00b2 10\u00b3 \u00bd \u2155 \u2460 \u2474",
    "roman numerals": "\u2160\u2161\u2162 \u2167 \u216c",
    "emoji": "ship it \U0001f680 \U0001f469\u200d\U0001f4bb done",
    "underscore": "snake_case __init__ _ __ a_b_c",
    "digits": "3.14 1,000 \u0663 \u0967\u0968",
    "mixed script": "caf\u00e9\u5de5\u7a0b\u5e08abc\u0434\u0435\u043b\u043e123",
    "combining after ascii": "e\u0301cole re\u0301sume\u0301",
    "lowercasing makes a combining mark": "I\u0307 \u0130",
    "title case letters": "\u01c5 \u01c8 \u01cb",
    "empty": "",
    "whitespace": "  \t\n\u00a0\u2003 ",
    "punctuation only": "!!! --- ... ###",
    "nul and controls": "a\x00b\x01c d\x7fe",
    "long": ("kubernetes distributed-systems C++ \u00dc\u00fc " * 200),
}


@pytest.mark.parametrize("name", TRICKY)
def test_tokenizer_matches_python_on_tricky_input(name):
    text = TRICKY[name]
    assert native.tokenize(text) == relevance.tokenize(text)


def test_ascii_fast_path_matches_python():
    """Pure-ASCII text takes a shortcut in the library; every ASCII character, alone and between letters."""
    for cp in range(128):
        for text in (chr(cp), f"a{chr(cp)}B", f"{chr(cp)}Zz9_"):
            assert native.tokenize(text) == relevance.tokenize(text), repr(text)
    everything = "".join(chr(cp) for cp in range(128)) * 3
    assert native.tokenize(everything) == relevance.tokenize(everything)


def test_tokenizer_known_values():
    assert native.tokenize("\u0130stanbul") == ["i", "stanbul"]  # i + U+0307, which is not \w
    assert native.tokenize("\u039f\u0394\u039f\u03a3 \u03a3\u03a3") == [
        "\u03bf\u03b4\u03bf\u03c2",
        "\u03c3\u03c2",
    ]
    assert native.tokenize("x\u00b2 \u2163") == ["x\u00b2", "\u2173"]
    assert native.tokenize("a\U0001f680b") == ["a", "b"]
    assert native.tokenize("") == []


def test_tokenizer_rejects_text_that_is_not_utf8_encodable():
    with pytest.raises(UnicodeEncodeError):
        native.tokenize("lone surrogate \ud800")


def test_tokenizer_matches_python_on_random_strings():
    """Seeded random strings built from assigned code points, weighted toward the cases lower() and \\w
    treat specially (sigma, dotted I, combining marks, numerics)."""
    rng = random.Random(20260101)
    py_unicode = _version(unicodedata.unidata_version)
    assert _version(native.unicode_versions()[0]) >= py_unicode
    assigned = [
        cp
        for cp in range(0x110000)
        if cp not in SURROGATES and unicodedata.category(chr(cp)) != "Cn"
    ]
    special = [
        ord(c)
        for c in "\u03a3\u03c3\u03c2\u0130\u0131\u0307\u0301 _-.\u00b2\u2162\u00df\u1e9e\u01c5"
    ]
    for _ in range(4000):
        n = rng.randint(1, 40)
        text = "".join(
            chr(rng.choice(special) if rng.random() < 0.4 else rng.choice(assigned))
            for _ in range(n)
        )
        assert native.tokenize(text) == relevance.tokenize(text), repr(text)


# --- corpora ------------------------------------------------------------------------------------

COMMON = "the engineer role team"  # in every document: idf = log(n / (n + 1)) < 0
PHRASES = [
    "python distributed systems kubernetes",
    "machine learning retrieval ranking embeddings",
    "react typescript frontend design systems",
    "postgres sql data pipelines airflow",
    "c++ rust low latency systems",
    "sales quota outreach crm marketing",
    "r\u00e9sum\u00e9 na\u00efve caf\u00e9 z\u00fcrich \u00fcber stra\u00dfe",
    "\u0130stanbul \u0131s\u0131 \u0130ZM\u0130R \u00fcretim m\u00fchendisi",
    "\u039c\u03b7\u03c7\u03b1\u03bd\u03b9\u03ba\u03cc\u03c2 \u039f\u0394\u039f\u03a3 \u03a3\u03a3\u0391 \u03bc\u03b7\u03c7\u03b1\u03bd\u03b9\u03ba\u03cc\u03c2",
    "\u0438\u043d\u0436\u0435\u043d\u0435\u0440 \u043f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u043d\u043e\u0433\u043e \u043e\u0431\u0435\u0441\u043f\u0435\u0447\u0435\u043d\u0438\u044f",
    "\u81ea\u7136\u8a00\u8a9e\u51e6\u7406 \u30a8\u30f3\u30b8\u30cb\u30a2 \u673a\u5668\u5b66\u4e60 \ud55c\uad6d\uc5b4",
    "\u0645\u0647\u0646\u062f\u0633 \u0627\u0644\u0628\u0631\u0645\u062c\u064a\u0627\u062a \u0661\u0662\u0663",
    "\u0907\u0902\u091c\u0940\u0928\u093f\u092f\u0930 \u0915\u094d\u0937 \u0939\u093f\u0928\u094d\u0926\u0940",
    "x\u00b2 10\u00b3 \u00bd \u2162 \u2167 snake_case __init__ \U0001f680",
]


def fixture_documents(seed: int = 7, n: int = 260) -> list[tuple[int, str]]:
    """(id, similarity text) pairs in a shuffled, non-contiguous id order: topic-clustered postings, about a
    third non-ASCII, exact and near duplicates (ties), a document repeated twice (same tf), documents with no
    tokens, and COMMON words in every document (idf < 0)."""
    rng = random.Random(seed)
    pool = [w for phrase in PHRASES for w in phrase.split()]
    docs: list[str] = []
    for i in range(n):
        topic = PHRASES[i % len(PHRASES)].split()
        words = [rng.choice(topic) for _ in range(rng.randint(8, 30))]
        words += [rng.choice(pool) for _ in range(rng.randint(0, 6))]
        rng.shuffle(words)
        docs.append(f"{COMMON} {' '.join(words)}")
    docs += [
        docs[3],
        docs[3],
        docs[3],
        docs[10] + " extra",
        docs[10] + " other",
    ]  # ties and near ties
    docs += [f"{docs[20]} {docs[20]}"]  # same tf as docs[20], different counts
    docs += ["", "   ", "!!! --- ...", "\u0307\u0301"]  # zero tokens
    docs += ["unique"]  # a term in one document only
    rng.shuffle(docs)
    ids = rng.sample(range(10, 10 * len(docs)), len(docs))
    return [
        (pid, relevance._similarity_text(f"Title {pid % 7}", text)) for pid, text in zip(ids, docs)
    ]


def _assert_parity(result: bench_parity.ParityResult) -> None:
    detail = [(m.posting_id, m.kind, m.python[:3], m.native[:3]) for m in result.mismatches[:3]]
    assert result.ok, (
        f"{len(result.mismatches)} of {result.n_targets} targets differ; first: {detail}"
    )
    assert result.max_abs_score_diff <= bench_parity.SCORE_TOLERANCE


def test_fixture_corpus_every_top10_matches_python():
    documents = fixture_documents()
    assert sum(1 for _, t in documents if not t.isascii()) > len(documents) // 4
    result = bench_parity.compare(documents, k=10)
    assert result.n_targets == len(documents)
    _assert_parity(result)
    # Exercised, not vacuous: terms with idf < 0 (in every document) and tied scores both occur.
    token_lists = [relevance.tokenize(text) for _, text in documents]
    df = relevance.document_frequencies(token_lists)
    idf = relevance.inverse_document_frequencies(df, n_docs=len(documents))
    assert any(value < 0 for value in idf.values())
    vectors = relevance.python_similarity_vectors(documents)
    sims = [s for pid in list(vectors)[:40] for _, s in relevance.python_top_k(vectors, pid, 50)]
    assert len(set(sims)) < len(sims)


def test_fixture_corpus_deep_lists_match_python():
    """k larger than the corpus: the whole ranking, including every tail tie, is identical."""
    documents = fixture_documents(seed=11, n=120)
    _assert_parity(bench_parity.compare(documents, k=10_000, sample=25))


def test_zero_token_document_scores_zero_and_keeps_insertion_order():
    documents = [(5, "a b"), (1, ""), (9, "a b"), (3, "c")]
    index = native.Similarity(documents)
    assert index.top_k(1, 3) == [(5, 0.0), (9, 0.0), (3, 0.0)]
    assert [pid for pid, _ in index.top_k(5, 3) or []][0] == 9  # identical text ranks first
    assert index.top_k(404, 3) is None


def test_synthetic_corpus_matches_python():
    """The Phase 0 generator at ~3,000 postings (6% non-ASCII, 10% near-duplicates within a company).
    A seeded sample of query postings, because the Python reference is O(N) per query."""
    corpus = bench_corpus.generate_synthetic(3000)
    documents = [(p.id, relevance._similarity_text(p.title, p.text)) for p in corpus.postings]
    result = bench_parity.compare(documents, k=10, sample=60)
    _assert_parity(result)
    print(
        f"\nsynthetic N=3000: {result.n_targets} targets, {result.n_hits} hits, "
        f"max |diff| {result.max_abs_score_diff:.3g}, bit-identical scores {result.n_exact_score_matches}"
    )


# --- index behaviour ----------------------------------------------------------------------------


def test_top_k_limits():
    index = native.Similarity([(1, "a b"), (2, "a c"), (3, "b c")])
    assert len(index) == 3
    assert index.top_k(1, 0) == []
    assert len(index.top_k(1, 10**9) or []) == 2  # capped at the other documents
    with pytest.raises(ValueError):
        index.top_k(1, -1)


def test_invalid_input_raises_with_library_error():
    with pytest.raises(NativeError, match="duplicate document id 1"):
        native.Similarity([(1, "a"), (1, "b")])


# --- the parity harness itself (so a failing report can be trusted) ---------------------------------


def test_parity_result_classifies_mismatches():
    near = bench_parity.Mismatch(1, "ids", [(2, 0.5), (3, 0.5)], [(3, 0.5 + 1e-12), (2, 0.5)])
    far = bench_parity.Mismatch(2, "ids", [(2, 0.5), (3, 0.4)], [(3, 0.45), (2, 0.4)])
    off = bench_parity.Mismatch(3, "score", [(2, 0.5)], [(2, 0.6)])
    result = bench_parity.ParityResult(n_docs=9, n_targets=3, k=2, mismatches=[near, far, off])
    assert not result.ok
    assert (result.n_id_mismatch, result.n_near_tie_reorders, result.n_score_mismatch) == (2, 1, 1)


def test_parity_snapshot_report_renders_pass_and_fail():
    import parity_snapshot

    corpus = bench_corpus.generate_synthetic(30, seed=3)
    documents = [(p.id, relevance._similarity_text(p.title, p.text)) for p in corpus.postings]
    result = bench_parity.compare(documents, k=5)
    versions = {"ICU": "x"}
    assert "**Result: PASS.**" in parity_snapshot.render(
        corpus, documents, result, 1.0, versions, 0
    )
    result.mismatches.append(bench_parity.Mismatch(7, "ids", [(1, 0.5)], [(2, 0.5)]))
    report = parity_snapshot.render(corpus, documents, result, 1.0, versions, 0)
    assert "**Result: FAIL.**" in report and "posting 7 (ids)" in report
