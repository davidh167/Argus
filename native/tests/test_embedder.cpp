// Needs a GGUF embedding model. Set ARGUS_MODEL, or the Ollama nomic-embed-text blob is used if
// present; otherwise these tests are skipped (they are not failures).
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

#include "argus.h"

namespace {

std::string model_path() {
    if (const char* p = std::getenv("ARGUS_MODEL")) return p;
    const char* home = std::getenv("HOME");
    if (home == nullptr) return {};
    const std::filesystem::path blob = std::filesystem::path(home) / ".ollama/models/blobs" /
        "sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6";  // nomic-embed-text:latest
    return std::filesystem::exists(blob) ? blob.string() : std::string{};
}

struct Embedder {
    argus_embedder* e = nullptr;
    size_t dim = 0;
    Embedder() {
        const auto path = model_path();
        if (path.empty()) SKIP("no embedding model available (set ARGUS_MODEL)");
        REQUIRE(argus_embedder_open(path.c_str(), 0, &e) == ARGUS_OK);
        REQUIRE(argus_embedder_dim(e, &dim) == ARGUS_OK);
    }
    ~Embedder() { argus_embedder_close(e); }
    Embedder(const Embedder&) = delete;
    Embedder& operator=(const Embedder&) = delete;

    std::vector<float> embed(const std::vector<std::string>& texts, std::vector<uint8_t>* trunc = nullptr) {
        std::vector<const char*> ptrs;
        std::vector<size_t> lens;
        for (const auto& t : texts) {
            ptrs.push_back(t.data());
            lens.push_back(t.size());
        }
        std::vector<float> out(texts.size() * dim);
        if (trunc) trunc->assign(texts.size(), 0);
        REQUIRE(argus_embed_batch(e, ptrs.data(), lens.data(), texts.size(), out.data(), dim, trunc ? trunc->data() : nullptr) ==
                ARGUS_OK);
        return out;
    }
};

double dot(const float* a, const float* b, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += static_cast<double>(a[i]) * b[i];
    return s;
}

}  // namespace

TEST_CASE("vectors have the model's dimension and unit length") {
    Embedder m;
    CHECK(m.dim == 768);
    const auto v = m.embed({"distributed systems design"});
    CHECK(dot(v.data(), v.data(), m.dim) == Catch::Approx(1.0).margin(1e-5));
}

TEST_CASE("embedding is deterministic and a batch matches one-at-a-time") {
    Embedder m;
    const std::vector<std::string> texts = {"PyTorch", "C++", "retrieval augmented generation pipelines", "Kubernetes"};
    const auto batch = m.embed(texts);
    for (size_t i = 0; i < texts.size(); ++i) {
        const auto single = m.embed({texts[i]});
        CHECK(dot(single.data(), &batch[i * m.dim], m.dim) > 0.9999);
    }
    const auto again = m.embed(texts);
    for (size_t i = 0; i < batch.size(); ++i) REQUIRE(again[i] == batch[i]);
}

TEST_CASE("related phrases are closer than unrelated ones") {
    Embedder m;
    const auto v = m.embed({"machine learning", "deep learning", "pastry baking"});
    CHECK(dot(&v[0], &v[m.dim], m.dim) > dot(&v[0], &v[2 * m.dim], m.dim));
}

TEST_CASE("more texts than one group holds are all embedded, in order") {
    Embedder m;
    std::vector<std::string> texts;
    for (int i = 0; i < 40; ++i) texts.push_back("skill number " + std::to_string(i));
    const auto batch = m.embed(texts);
    for (int i : {0, 17, 39}) {
        const auto single = m.embed({texts[static_cast<size_t>(i)]});
        CHECK(dot(single.data(), &batch[static_cast<size_t>(i) * m.dim], m.dim) > 0.9999);
    }
}

TEST_CASE("over-long input is truncated and flagged, short input is not") {
    Embedder m;
    std::string long_text;
    for (int i = 0; i < 6000; ++i) long_text += "word" + std::to_string(i % 50) + " ";
    std::vector<uint8_t> trunc;
    const auto v = m.embed({long_text, "short"}, &trunc);
    CHECK(trunc[0] == 1);
    CHECK(trunc[1] == 0);
    CHECK(dot(&v[0], &v[0], m.dim) == Catch::Approx(1.0).margin(1e-5));
}

TEST_CASE("empty text embeds, bad arguments are rejected") {
    Embedder m;
    CHECK(m.embed({""}).size() == m.dim);
    float out[8];
    CHECK(argus_embed_batch(nullptr, nullptr, nullptr, 0, out, 8, nullptr) == ARGUS_ERR_ARG);
    const char* t = "x";
    const size_t l = 1;
    CHECK(argus_embed_batch(m.e, &t, &l, 1, out, 8, nullptr) == ARGUS_ERR_ARG);  // wrong dim
    argus_embedder* none = nullptr;
    CHECK(argus_embedder_open("/no/such/model.gguf", 0, &none) == ARGUS_ERR_MODEL);
    CHECK(none == nullptr);
    CHECK(argus_embedder_open(nullptr, 0, &none) == ARGUS_ERR_ARG);
    argus_embedder_close(nullptr);
}
