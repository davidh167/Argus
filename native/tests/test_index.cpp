#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "argus.h"

namespace {

struct Index {
    argus_handle* h = nullptr;
    explicit Index(const char* path = ":memory:") { REQUIRE(argus_open(path, &h) == ARGUS_OK); }
    ~Index() { argus_close(h); }
    Index(const Index&) = delete;
    Index& operator=(const Index&) = delete;

    void add(int64_t id, const std::string& text) { REQUIRE(argus_upsert(h, id, text.data(), text.size()) == ARGUS_OK); }

    std::vector<argus_hit> search(const std::string& q, size_t k = 10) {
        std::vector<argus_hit> hits(k);
        size_t n = 0;
        REQUIRE(argus_search_bm25(h, q.data(), q.size(), k, hits.data(), &n) == ARGUS_OK);
        hits.resize(n);
        return hits;
    }
};

}  // namespace

TEST_CASE("search ranks the more relevant document first") {
    Index idx;
    idx.add(1, "python machine learning engineer building retrieval systems");
    idx.add(2, "sales manager for enterprise accounts");
    idx.add(3, "python python python scripting");
    const auto hits = idx.search("python retrieval");
    REQUIRE(hits.size() == 2);  // doc 2 matches no term and is not returned
    CHECK(hits[0].id == 1);
    CHECK(hits[0].score > hits[1].score);
}

TEST_CASE("upsert replaces and remove deletes") {
    Index idx;
    idx.add(7, "alpha");
    idx.add(7, "beta");  // same id: replaced
    size_t n = 0;
    REQUIRE(argus_count(idx.h, &n) == ARGUS_OK);
    CHECK(n == 1);
    CHECK(idx.search("alpha").empty());
    CHECK(idx.search("beta").size() == 1);
    REQUIRE(argus_remove(idx.h, 7) == ARGUS_OK);
    CHECK(idx.search("beta").empty());
    REQUIRE(argus_remove(idx.h, 12345) == ARGUS_OK);  // removing a missing id is not an error
}

TEST_CASE("respects the result capacity") {
    Index idx;
    for (int i = 0; i < 20; ++i) idx.add(i, "shared term " + std::to_string(i));
    CHECK(idx.search("shared", 5).size() == 5);
    CHECK(idx.search("shared", 100).size() == 20);
    CHECK(idx.search("shared", 0).empty());
}

TEST_CASE("free text with FTS5 syntax characters never raises a query error") {
    Index idx;
    idx.add(1, "c++ and rust systems programming, 3x faster");
    for (const char* q : {"c++", "\"unbalanced", "AND OR NOT", "(parens", "col:val", "foo*", "-minus", "a \"quoted\" b",
                          "&", "", "   ", "NEAR(x y)"}) {
        std::vector<argus_hit> hits(10);
        size_t n = 0;
        INFO("query: " << q);
        CHECK(argus_search_bm25(idx.h, q, std::char_traits<char>::length(q), 10, hits.data(), &n) == ARGUS_OK);
    }
    CHECK(idx.search("systems").size() == 1);
}

TEST_CASE("unicode: case and accent folding, punctuation separates") {
    Index idx;
    idx.add(1, "RÉSUMÉ parsing for café owners");
    idx.add(2, "company’s AI—driven platform");
    CHECK(idx.search("resume").size() == 1);   // accent- and case-insensitive
    CHECK(idx.search("CAFE").size() == 1);
    CHECK(idx.search("driven").size() == 1);   // an em dash separates words
    CHECK(idx.search("ai").front().id == 2);
}

TEST_CASE("bad arguments are rejected, not crashed on") {
    argus_handle* h = nullptr;
    CHECK(argus_open(nullptr, &h) == ARGUS_ERR_ARG);
    CHECK(argus_open(":memory:", nullptr) == ARGUS_ERR_ARG);
    CHECK(argus_open("/nonexistent-dir/x/y.db", &h) == ARGUS_ERR_DB);
    CHECK(h == nullptr);
    CHECK(argus_upsert(nullptr, 1, "a", 1) == ARGUS_ERR_ARG);
    size_t n = 0;
    CHECK(argus_count(nullptr, &n) == ARGUS_ERR_ARG);
    argus_close(nullptr);  // no-op
    CHECK(std::string(argus_last_error(nullptr)).empty());

    Index idx;
    CHECK(argus_search_bm25(idx.h, "x", 1, 5, nullptr, &n) == ARGUS_ERR_ARG);
    CHECK(argus_search_bm25(idx.h, "x", 1, 5, nullptr, nullptr) == ARGUS_ERR_ARG);
}

TEST_CASE("a persistent index survives close and reopen") {
    const auto path = std::filesystem::temp_directory_path() / "argus_test_index.db";
    std::filesystem::remove(path);
    {
        Index idx(path.c_str());
        idx.add(1, "persistent document about retrieval");
    }
    {
        Index idx(path.c_str());
        CHECK(idx.search("retrieval").size() == 1);
    }
    std::filesystem::remove(path);
    std::filesystem::remove(path.string() + "-wal");
    std::filesystem::remove(path.string() + "-shm");
}
