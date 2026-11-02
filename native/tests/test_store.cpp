// The index used as a persistent key -> (text, vector) store: argus_get_vector, argus_get_text,
// argus_meta_get / argus_meta_set.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "argus.h"

namespace {

constexpr size_t kDim = 4;

struct Index {
    argus_handle* h = nullptr;
    explicit Index(const char* path = ":memory:", size_t dim = kDim) {
        REQUIRE((dim == 0 ? argus_open(path, &h) : argus_open_with_vectors(path, dim, &h)) == ARGUS_OK);
    }
    ~Index() { argus_close(h); }
    Index(const Index&) = delete;
    Index& operator=(const Index&) = delete;
};

// A database file in the temp directory, with its WAL files, removed before and after.
struct TempDb {
    std::filesystem::path path;
    explicit TempDb(const char* name) : path(std::filesystem::temp_directory_path() / name) { clean(); }
    ~TempDb() { clean(); }
    void clean() const {
        std::filesystem::remove(path);
        std::filesystem::remove(path.string() + "-wal");
        std::filesystem::remove(path.string() + "-shm");
    }
};

std::string read_text(argus_handle* h, int64_t id, size_t cap, size_t* len, int* found) {
    std::string buf(cap, '\0');
    REQUIRE(argus_get_text(h, id, buf.data(), cap, len, found) == ARGUS_OK);
    buf.resize(std::min(cap, *len));
    return buf;
}

std::string read_meta(argus_handle* h, const char* key, int* found) {
    std::string buf(64, '\0');
    size_t len = 0;
    REQUIRE(argus_meta_get(h, key, buf.data(), buf.size(), &len, found) == ARGUS_OK);
    buf.resize(std::min<size_t>(buf.size(), len));
    return buf;
}

}  // namespace

TEST_CASE("a stored vector reads back bit-exact, also after reopening the file") {
    TempDb db("argus_getvec_test.db");
    const std::vector<float> v{0.1f, -0.333333343f, 1e-30f, 7.0f};
    {
        Index idx(db.path.c_str());
        REQUIRE(argus_upsert_vector(idx.h, 42, v.data(), v.size()) == ARGUS_OK);
    }
    Index idx(db.path.c_str());
    std::vector<float> out(kDim, 0.0f);
    int found = 0;
    REQUIRE(argus_get_vector(idx.h, 42, out.data(), kDim, &found) == ARGUS_OK);
    CHECK(found == 1);
    CHECK(std::memcmp(out.data(), v.data(), kDim * sizeof(float)) == 0);
    REQUIRE(argus_get_vector(idx.h, 43, out.data(), kDim, &found) == ARGUS_OK);
    CHECK(found == 0);
}

TEST_CASE("argus_get_vector rejects bad arguments") {
    Index idx;
    std::vector<float> out(kDim);
    int found = 0;
    CHECK(argus_get_vector(nullptr, 1, out.data(), kDim, &found) == ARGUS_ERR_ARG);
    CHECK(argus_get_vector(idx.h, 1, nullptr, kDim, &found) == ARGUS_ERR_ARG);
    CHECK(argus_get_vector(idx.h, 1, out.data(), kDim, nullptr) == ARGUS_ERR_ARG);
    CHECK(argus_get_vector(idx.h, 1, out.data(), kDim + 1, &found) == ARGUS_ERR_ARG);
    Index plain(":memory:", 0);
    CHECK(argus_get_vector(plain.h, 1, out.data(), kDim, &found) == ARGUS_ERR_ARG);
}

TEST_CASE("stored text reads back exactly, including UTF-8, empty text and short buffers") {
    Index idx;
    const std::string utf8 = "caf\xC3\xA9 \xE6\x97\xA5\xE6\x9C\xAC \"quoted\"";
    REQUIRE(argus_upsert(idx.h, 7, utf8.data(), utf8.size()) == ARGUS_OK);
    REQUIRE(argus_upsert(idx.h, 8, "", 0) == ARGUS_OK);
    size_t len = 0;
    int found = 0;

    CHECK(read_text(idx.h, 7, 256, &len, &found) == utf8);
    CHECK(found == 1);
    CHECK(len == utf8.size());

    // A buffer that is too small gets a prefix, and *len says how much room is needed.
    CHECK(read_text(idx.h, 7, 4, &len, &found) == utf8.substr(0, 4));
    CHECK(found == 1);
    CHECK(len == utf8.size());

    read_text(idx.h, 8, 16, &len, &found);
    CHECK(found == 1);
    CHECK(len == 0);

    read_text(idx.h, 9, 16, &len, &found);
    CHECK(found == 0);

    char c;
    CHECK(argus_get_text(nullptr, 7, &c, 1, &len, &found) == ARGUS_ERR_ARG);
    CHECK(argus_get_text(idx.h, 7, nullptr, 1, &len, &found) == ARGUS_ERR_ARG);
    CHECK(argus_get_text(idx.h, 7, &c, 1, nullptr, &found) == ARGUS_ERR_ARG);
    CHECK(argus_get_text(idx.h, 7, &c, 1, &len, nullptr) == ARGUS_ERR_ARG);
}

TEST_CASE("metadata persists, replaces, and a user key named dim does not touch the dimension") {
    TempDb db("argus_meta_test.db");
    int found = 0;
    {
        Index idx(db.path.c_str());
        read_meta(idx.h, "model", &found);
        CHECK(found == 0);
        REQUIRE(argus_meta_set(idx.h, "model", "abc") == ARGUS_OK);
        REQUIRE(argus_meta_set(idx.h, "model", "abcd") == ARGUS_OK);
        REQUIRE(argus_meta_set(idx.h, "dim", "999") == ARGUS_OK);
    }
    // Reopening with the real dimension still works, so the library's own "dim" key was not clobbered.
    Index idx(db.path.c_str());
    CHECK(read_meta(idx.h, "model", &found) == "abcd");
    CHECK(found == 1);
    CHECK(read_meta(idx.h, "dim", &found) == "999");
}

TEST_CASE("metadata calls reject bad arguments and work on an index without vectors") {
    Index idx(":memory:", 0);
    char buf[8];
    size_t len = 0;
    int found = 0;
    CHECK(argus_meta_set(nullptr, "k", "v") == ARGUS_ERR_ARG);
    CHECK(argus_meta_set(idx.h, nullptr, "v") == ARGUS_ERR_ARG);
    CHECK(argus_meta_set(idx.h, "k", nullptr) == ARGUS_ERR_ARG);
    CHECK(argus_meta_get(idx.h, "k", buf, sizeof buf, nullptr, &found) == ARGUS_ERR_ARG);
    REQUIRE(argus_meta_set(idx.h, "k", "v") == ARGUS_OK);
    REQUIRE(argus_meta_get(idx.h, "k", buf, sizeof buf, &len, &found) == ARGUS_OK);
    CHECK(found == 1);
    CHECK(len == 1);
    CHECK(buf[0] == 'v');
}

TEST_CASE("a batch of records is stored, replaces by id, and is searchable by text and by vector") {
    TempDb db("argus_records_test.db");
    const std::vector<int64_t> ids{10, 20, 30};
    const std::vector<std::string> texts{"alpha beta", "gamma", ""};
    const std::vector<float> vecs{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    std::vector<const char*> ptrs;
    std::vector<size_t> lens;
    for (const auto& t : texts) {
        ptrs.push_back(t.data());
        lens.push_back(t.size());
    }
    {
        Index idx(db.path.c_str());
        REQUIRE(argus_upsert_records(idx.h, 3, ids.data(), ptrs.data(), lens.data(), vecs.data(), kDim) == ARGUS_OK);
        // Replacing id 20 changes both its text and its vector.
        const int64_t id = 20;
        const char* text = "delta";
        const size_t len = 5;
        const std::vector<float> v{0, 0, 0, 1};
        REQUIRE(argus_upsert_records(idx.h, 1, &id, &text, &len, v.data(), kDim) == ARGUS_OK);
        REQUIRE(argus_upsert_records(idx.h, 0, nullptr, nullptr, nullptr, nullptr, kDim) == ARGUS_OK);
    }
    Index idx(db.path.c_str());
    size_t len = 0;
    int found = 0;
    CHECK(read_text(idx.h, 10, 64, &len, &found) == "alpha beta");
    CHECK(read_text(idx.h, 20, 64, &len, &found) == "delta");
    read_text(idx.h, 30, 64, &len, &found);
    CHECK((found == 1 && len == 0));

    std::vector<float> out(kDim);
    REQUIRE(argus_get_vector(idx.h, 20, out.data(), kDim, &found) == ARGUS_OK);
    CHECK(out == std::vector<float>{0, 0, 0, 1});

    argus_hit hit{};
    size_t n = 0;
    REQUIRE(argus_search_bm25(idx.h, "beta", 4, 1, &hit, &n) == ARGUS_OK);
    REQUIRE(n == 1);
    CHECK(hit.id == 10);
    const std::vector<float> q{0, 0, 1, 0};
    REQUIRE(argus_search_dense(idx.h, q.data(), kDim, 1, &hit, &n) == ARGUS_OK);
    CHECK(hit.id == 30);
}

TEST_CASE("argus_upsert_records rejects bad arguments and stores nothing") {
    Index idx;
    const int64_t id = 1;
    const char* text = "x";
    const size_t len = 1;
    const std::vector<float> v{1, 0, 0, 0};
    CHECK(argus_upsert_records(nullptr, 1, &id, &text, &len, v.data(), kDim) == ARGUS_ERR_ARG);
    CHECK(argus_upsert_records(idx.h, 1, &id, &text, &len, v.data(), kDim + 1) == ARGUS_ERR_ARG);
    CHECK(argus_upsert_records(idx.h, 1, nullptr, &text, &len, v.data(), kDim) == ARGUS_ERR_ARG);
    CHECK(argus_upsert_records(idx.h, 1, &id, &text, &len, nullptr, kDim) == ARGUS_ERR_ARG);
    const char* null_text = nullptr;
    CHECK(argus_upsert_records(idx.h, 1, &id, &null_text, &len, v.data(), kDim) == ARGUS_ERR_ARG);
    Index plain(":memory:", 0);
    CHECK(argus_upsert_records(plain.h, 1, &id, &text, &len, v.data(), kDim) == ARGUS_ERR_ARG);
    size_t count = 99;
    REQUIRE(argus_count(idx.h, &count) == ARGUS_OK);
    CHECK(count == 0);
}

TEST_CASE("vectors stored before and after the first search are all searchable") {
    TempDb db("argus_lazy_graph_test.db");
    const std::vector<float> a{1, 0, 0, 0}, b{0, 1, 0, 0}, c{0, 0, 1, 0};
    const auto nearest = [](argus_handle* h, const std::vector<float>& q) {
        argus_hit hit{};
        size_t n = 0;
        REQUIRE(argus_search_dense(h, q.data(), q.size(), 1, &hit, &n) == ARGUS_OK);
        REQUIRE(n == 1);
        return hit.id;
    };
    {
        Index idx(db.path.c_str());
        REQUIRE(argus_upsert_vector(idx.h, 1, a.data(), kDim) == ARGUS_OK);
        CHECK(nearest(idx.h, a) == 1);  // builds the graph
        REQUIRE(argus_upsert_vector(idx.h, 2, b.data(), kDim) == ARGUS_OK);  // goes into the built graph
        CHECK(nearest(idx.h, b) == 2);
    }
    Index idx(db.path.c_str());
    REQUIRE(argus_upsert_vector(idx.h, 3, c.data(), kDim) == ARGUS_OK);  // graph not built yet
    CHECK(nearest(idx.h, a) == 1);
    CHECK(nearest(idx.h, b) == 2);
    CHECK(nearest(idx.h, c) == 3);
}
