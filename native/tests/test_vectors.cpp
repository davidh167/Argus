#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "argus.h"

namespace {

constexpr size_t kDim = 4;

std::vector<float> unit(float a, float b, float c, float d) {
    std::vector<float> v{a, b, c, d};
    float n = 0;
    for (float x : v) n += x * x;
    n = std::sqrt(n);
    for (float& x : v) x /= n;
    return v;
}

struct VecIndex {
    argus_handle* h = nullptr;
    explicit VecIndex(const char* path = ":memory:", size_t dim = kDim) { REQUIRE(argus_open_with_vectors(path, dim, &h) == ARGUS_OK); }
    ~VecIndex() { argus_close(h); }
    VecIndex(const VecIndex&) = delete;
    VecIndex& operator=(const VecIndex&) = delete;

    void text(int64_t id, const std::string& t) { REQUIRE(argus_upsert(h, id, t.data(), t.size()) == ARGUS_OK); }
    void vec(int64_t id, const std::vector<float>& v) { REQUIRE(argus_upsert_vector(h, id, v.data(), v.size()) == ARGUS_OK); }

    std::vector<argus_hit> dense(const std::vector<float>& q, size_t k = 10) {
        std::vector<argus_hit> hits(k);
        size_t n = 0;
        REQUIRE(argus_search_dense(h, q.data(), q.size(), k, hits.data(), &n) == ARGUS_OK);
        hits.resize(n);
        return hits;
    }
    std::vector<argus_hit> hybrid(const std::string& q, const std::vector<float>& qv, size_t k = 10) {
        std::vector<argus_hit> hits(k);
        size_t n = 0;
        REQUIRE(argus_search_hybrid(h, q.data(), q.size(), qv.data(), qv.size(), k, hits.data(), &n) == ARGUS_OK);
        hits.resize(n);
        return hits;
    }
};

}  // namespace

TEST_CASE("dense search ranks by cosine similarity") {
    VecIndex idx;
    idx.vec(1, unit(1, 0, 0, 0));
    idx.vec(2, unit(1, 1, 0, 0));
    idx.vec(3, unit(0, 0, 1, 0));
    const auto hits = idx.dense(unit(1, 0.1f, 0, 0));
    REQUIRE(hits.size() == 3);
    CHECK(hits[0].id == 1);
    CHECK(hits[1].id == 2);
    CHECK(hits[2].id == 3);
    CHECK(hits[0].score == Catch::Approx(0.995).margin(0.01));
    CHECK(hits[2].score == Catch::Approx(0.0).margin(1e-3));
}

TEST_CASE("replacing and removing vectors updates the index") {
    VecIndex idx;
    idx.vec(1, unit(1, 0, 0, 0));
    idx.vec(2, unit(0, 1, 0, 0));
    idx.vec(1, unit(0, 0, 1, 0));  // replace
    CHECK(idx.dense(unit(0, 0, 1, 0), 1).front().id == 1);
    CHECK(idx.dense(unit(1, 0, 0, 0), 1).front().id != 1);
    REQUIRE(argus_remove(idx.h, 1) == ARGUS_OK);
    for (const auto& hit : idx.dense(unit(0, 0, 1, 0))) CHECK(hit.id != 1);
}

TEST_CASE("reopening a file rebuilds the vector index from storage") {
    const auto path = std::filesystem::temp_directory_path() / "argus_vec_test.db";
    std::filesystem::remove(path);
    {
        VecIndex idx(path.c_str());
        idx.vec(5, unit(1, 0, 0, 0));
        idx.vec(6, unit(0, 1, 0, 0));
        idx.text(5, "persisted document");
    }
    {
        VecIndex idx(path.c_str());
        CHECK(idx.dense(unit(0, 1, 0, 0), 1).front().id == 6);
        CHECK(idx.hybrid("persisted", unit(1, 0, 0, 0)).front().id == 5);
    }
    argus_handle* h = nullptr;
    CHECK(argus_open_with_vectors(path.c_str(), kDim + 1, &h) == ARGUS_ERR_ARG);  // wrong dimension for this file
    CHECK(h == nullptr);
    std::filesystem::remove(path);
    std::filesystem::remove(path.string() + "-wal");
    std::filesystem::remove(path.string() + "-shm");
}

TEST_CASE("hybrid search fuses the keyword and dense rankings") {
    VecIndex idx;
    // Keyword ranking: 1 (strongest), 3 (weaker: long text dilutes it), 2 (no match).
    // Dense ranking for the query vector: 3 (identical), 2 (close), 1 (orthogonal).
    // RRF: doc 3 = 1/62 + 1/61 beats doc 1 = 1/61 + 1/63 and doc 2 = 1/62 alone.
    idx.text(1, "retrieval retrieval retrieval");
    idx.text(2, "cooking recipes and baking");
    idx.text(3, "retrieval systems for large scale enterprise search and many other unrelated words here");
    idx.vec(1, unit(0, 0, 0, 1));
    idx.vec(2, unit(0.9f, 0.3f, 0, 0));
    idx.vec(3, unit(1, 0, 0, 0));
    const auto hits = idx.hybrid("retrieval", unit(1, 0, 0, 0));
    REQUIRE(hits.size() == 3);
    CHECK(hits[0].id == 3);
    CHECK(hits[1].id == 1);
    CHECK(hits[2].id == 2);
    CHECK(hits[0].score == Catch::Approx(1.0 / 62 + 1.0 / 61));
}

TEST_CASE("hybrid with no keyword matches still returns dense results") {
    VecIndex idx;
    idx.text(1, "alpha");
    idx.vec(1, unit(1, 0, 0, 0));
    const auto hits = idx.hybrid("zzzz", unit(1, 0, 0, 0));
    REQUIRE(hits.size() == 1);
    CHECK(hits[0].id == 1);
}

TEST_CASE("vector calls reject bad arguments and indexes opened without vectors") {
    VecIndex idx;
    const auto v = unit(1, 0, 0, 0);
    size_t n = 0;
    argus_hit hit;
    CHECK(argus_upsert_vector(idx.h, 1, v.data(), kDim + 1) == ARGUS_ERR_ARG);
    CHECK(argus_search_dense(idx.h, v.data(), kDim + 1, 1, &hit, &n) == ARGUS_ERR_ARG);
    CHECK(argus_upsert_vector(nullptr, 1, v.data(), kDim) == ARGUS_ERR_ARG);
    CHECK(argus_upsert_vector(idx.h, 1, nullptr, kDim) == ARGUS_ERR_ARG);
    CHECK(argus_search_hybrid(idx.h, "a", 1, nullptr, kDim, 1, &hit, &n) == ARGUS_ERR_ARG);

    argus_handle* plain = nullptr;
    REQUIRE(argus_open(":memory:", &plain) == ARGUS_OK);
    CHECK(argus_upsert_vector(plain, 1, v.data(), kDim) == ARGUS_ERR_ARG);  // no vector index here
    argus_close(plain);

    argus_handle* none = nullptr;
    CHECK(argus_open_with_vectors(":memory:", 0, &none) == ARGUS_ERR_ARG);
}

TEST_CASE("an empty vector index answers queries with no hits") {
    VecIndex idx;
    CHECK(idx.dense(unit(1, 0, 0, 0)).empty());
    CHECK(idx.hybrid("anything", unit(1, 0, 0, 0)).empty());
}

TEST_CASE("many vectors grow the index past its initial capacity") {
    VecIndex idx;
    for (int i = 0; i < 3000; ++i) idx.vec(i, unit(1.0f + static_cast<float>(i % 7), static_cast<float>(i % 13), static_cast<float>(i % 5), 1.0f));
    CHECK(idx.dense(unit(1, 0, 0, 1), 5).size() == 5);
}
