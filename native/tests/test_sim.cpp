// Sparse TF-IDF similarity (argus_sim_*) and its tokenizer hooks. The exhaustive agreement with Python's
// \w and str.lower() is checked from Python (tests/test_native_similarity.py); these tests cover the C API
// contract and a few hand-computed values.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>
#include <vector>

#include "argus.h"

namespace {

struct Sim {
    argus_sim* s = nullptr;
    Sim() { REQUIRE(argus_sim_create(&s) == ARGUS_OK); }
    ~Sim() { argus_sim_free(s); }
    Sim(const Sim&) = delete;
    Sim& operator=(const Sim&) = delete;

    argus_status add(int64_t id, const std::string& text) { return argus_sim_add(s, id, text.data(), text.size()); }
    std::vector<argus_hit> topk(int64_t id, size_t k, int* found = nullptr) {
        std::vector<argus_hit> hits(k + 1);
        size_t n = 0;
        int f = 0;
        REQUIRE(argus_sim_topk(s, id, k, hits.data(), &n, &f) == ARGUS_OK);
        if (found) *found = f;
        hits.resize(n);
        return hits;
    }
};

std::vector<std::string> tokens(const std::string& text) {
    size_t need = 0;
    REQUIRE(argus_text_tokenize(text.data(), text.size(), nullptr, 0, &need) == ARGUS_OK);
    std::string buf(need, '\0');
    size_t len = 0;
    REQUIRE(argus_text_tokenize(text.data(), text.size(), buf.data(), buf.size(), &len) == ARGUS_OK);
    REQUIRE(len == need);
    std::vector<std::string> out;
    size_t start = 0;
    for (size_t i = 0; i < buf.size(); ++i) {
        if (buf[i] == '\n') {
            out.push_back(buf.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

}  // namespace

TEST_CASE("tokenizer: lowercases, splits on non-word characters, keeps underscores and digits", "[sim][tokenizer]") {
    CHECK(tokens("Hello, World! C++ foo_bar 42") == std::vector<std::string>{"hello", "world", "c", "foo_bar", "42"});
    CHECK(tokens("").empty());
    CHECK(tokens("  \t\n ").empty());
}

TEST_CASE("tokenizer: full Unicode lowercase and word classes match Python", "[sim][tokenizer]") {
    // U+0130 lowercases to i + U+0307; the combining dot is not a word character, so it splits the token.
    CHECK(tokens("\xC4\xB0stanbul") == std::vector<std::string>{"i", "stanbul"});
    CHECK(tokens("STRASSE \xC3\x9F") == std::vector<std::string>{"strasse", "\xC3\x9F"});
    // Final sigma: word-final capital sigma becomes U+03C2, a medial one U+03C3.
    CHECK(tokens("\xCE\x9F\xCE\x94\xCE\x9F\xCE\xA3") == std::vector<std::string>{"\xCE\xBF\xCE\xB4\xCE\xBF\xCF\x82"});
    CHECK(tokens("\xCE\xA3\xCE\xA3\xCE\x91") == std::vector<std::string>{"\xCF\x83\xCF\x83\xCE\xB1"});
    // Superscript two (No), Roman numeral (Nl, lowercased to U+2173), vulgar fraction (No) are word characters via isnumeric.
    CHECK(tokens("x\xC2\xB2 \xE2\x85\xA3 \xC2\xBD") == std::vector<std::string>{"x\xC2\xB2", "\xE2\x85\xB3", "\xC2\xBD"});
    // Emoji and punctuation are not.
    CHECK(tokens("a\xF0\x9F\x98\x80" "b") == std::vector<std::string>{"a", "b"});
}

TEST_CASE("tokenizer rejects invalid UTF-8", "[sim][tokenizer]") {
    const std::string bad = "ok \xFF\xFE";
    size_t len = 0;
    char buf[8];
    CHECK(argus_text_tokenize(bad.data(), bad.size(), buf, sizeof buf, &len) == ARGUS_ERR_ARG);
}

TEST_CASE("unicode probe reports word class and lowercase mapping", "[sim][tokenizer]") {
    int word = 0;
    uint32_t lower[4];
    size_t n = 0;
    REQUIRE(argus_unicode_probe(0x130, &word, lower, 4, &n) == ARGUS_OK);
    CHECK(word == 1);
    REQUIRE(n == 2);
    CHECK(lower[0] == 0x69);
    CHECK(lower[1] == 0x307);
    REQUIRE(argus_unicode_probe(0x20, &word, lower, 4, &n) == ARGUS_OK);
    CHECK(word == 0);
    CHECK(argus_unicode_probe(0x110000, &word, lower, 4, &n) == ARGUS_ERR_ARG);
    CHECK(std::string(argus_unicode_version()).size() >= 3);
    CHECK(std::string(argus_icu_version()).size() >= 2);
}

TEST_CASE("sim: matches a hand-computed tf-idf cosine", "[sim]") {
    // Five docs, df: a=3, b=2, c=2. idf = log(5/(1+df)): a -> log(5/4), b = c -> log(5/3).
    Sim sim;
    REQUIRE(sim.add(10, "a b") == ARGUS_OK);
    REQUIRE(sim.add(20, "a b b c") == ARGUS_OK);
    REQUIRE(sim.add(30, "a c") == ARGUS_OK);
    REQUIRE(sim.add(40, "d") == ARGUS_OK);
    REQUIRE(sim.add(50, "d e") == ARGUS_OK);
    REQUIRE(argus_sim_finalize(sim.s) == ARGUS_OK);

    const double ia = std::log(5.0 / 4.0), ib = std::log(5.0 / 3.0), ic = std::log(5.0 / 3.0);
    // doc 10: {a: .5*ia, b: .5*ib}; doc 20: {a: .25*ia, b: .5*ib, c: .25*ic}
    const double d10[] = {.5 * ia, .5 * ib}, d20[] = {.25 * ia, .5 * ib, .25 * ic};
    const double dot = d10[0] * d20[0] + d10[1] * d20[1];
    const double n10 = std::sqrt(d10[0] * d10[0] + d10[1] * d10[1]);
    const double n20 = std::sqrt(d20[0] * d20[0] + d20[1] * d20[1] + d20[2] * d20[2]);

    int found = 0;
    const auto hits = sim.topk(10, 5, &found);
    REQUIRE(found == 1);
    REQUIRE(hits.size() == 4);  // the query document is excluded
    CHECK(hits[0].id == 20);
    CHECK(hits[0].score == Catch::Approx(dot / (n10 * n20)).margin(1e-15));
    CHECK(hits[3].score == 0.0);  // docs 40 and 50 share nothing with doc 10
}

TEST_CASE("sim: ties keep insertion order, zero-token and unknown documents behave", "[sim]") {
    Sim sim;
    REQUIRE(sim.add(5, "rust cpp go") == ARGUS_OK);
    REQUIRE(sim.add(1, "") == ARGUS_OK);           // no tokens: empty vector, scores 0.0
    REQUIRE(sim.add(9, "rust cpp go") == ARGUS_OK);  // identical to 5
    REQUIRE(sim.add(7, "rust cpp go") == ARGUS_OK);
    REQUIRE(sim.add(3, "java") == ARGUS_OK);
    REQUIRE(argus_sim_finalize(sim.s) == ARGUS_OK);

    // "java" and the three identical docs: terms in every doc have idf <= 0, so some scores are 0.0 or
    // negative; whatever they are, equal scores must come out in insertion order.
    const auto hits = sim.topk(5, 10);
    REQUIRE(hits.size() == 4);
    CHECK(hits[0].id == 9);
    CHECK(hits[1].id == 7);
    for (size_t i = 1; i < hits.size(); ++i) CHECK(hits[i - 1].score >= hits[i].score);

    const auto from_empty = sim.topk(1, 3);  // an empty query vector scores 0.0 against all: insertion order
    REQUIRE(from_empty.size() == 3);
    CHECK(from_empty[0].id == 5);
    CHECK(from_empty[1].id == 9);
    CHECK(from_empty[2].id == 7);
    for (const auto& h : from_empty) CHECK(h.score == 0.0);

    int found = 1;
    CHECK(sim.topk(404, 3, &found).empty());
    CHECK(found == 0);
    CHECK(sim.topk(5, 0).empty());
}

TEST_CASE("sim: lifecycle and argument errors", "[sim]") {
    CHECK(argus_sim_create(nullptr) == ARGUS_ERR_ARG);
    Sim sim;
    REQUIRE(sim.add(1, "x y") == ARGUS_OK);
    CHECK(sim.add(1, "again") == ARGUS_ERR_ARG);
    CHECK(std::string(argus_sim_last_error(sim.s)).find("duplicate") != std::string::npos);
    const std::string bad = "\xC3\x28";
    CHECK(argus_sim_add(sim.s, 2, bad.data(), bad.size()) == ARGUS_ERR_ARG);

    argus_hit hit;
    size_t n = 0;
    int found = 0;
    CHECK(argus_sim_topk(sim.s, 1, 1, &hit, &n, &found) == ARGUS_ERR_ARG);  // before finalize
    REQUIRE(argus_sim_finalize(sim.s) == ARGUS_OK);
    CHECK(argus_sim_finalize(sim.s) == ARGUS_ERR_ARG);
    CHECK(sim.add(3, "late") == ARGUS_ERR_ARG);
    size_t count = 0;
    REQUIRE(argus_sim_count(sim.s, &count) == ARGUS_OK);
    CHECK(count == 1);  // the rejected adds left nothing behind
    CHECK(argus_sim_topk(nullptr, 1, 1, &hit, &n, &found) == ARGUS_ERR_ARG);
    CHECK(argus_sim_topk(sim.s, 1, 1, nullptr, &n, &found) == ARGUS_ERR_ARG);
}
