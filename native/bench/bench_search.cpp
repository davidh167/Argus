// Synthetic-corpus benchmark so it runs without private data. Real-corpus comparison
// against bm25s is bench/compare_with_bm25s.py.
#include <benchmark/benchmark.h>

#include <random>
#include <string>
#include <vector>

#include "argus.h"

namespace {

std::vector<std::string> make_vocab(std::size_t n) {
    std::vector<std::string> v;
    for (std::size_t i = 0; i < n; ++i) v.push_back("w" + std::to_string(i));
    return v;
}

// Zipf-ish term frequencies: low-numbered terms are common, like natural text.
std::string make_doc(std::mt19937& rng, const std::vector<std::string>& vocab, std::size_t words) {
    std::geometric_distribution<std::size_t> pick(0.002);
    std::string doc;
    for (std::size_t i = 0; i < words; ++i) {
        doc += vocab[pick(rng) % vocab.size()];
        doc.push_back(' ');
    }
    return doc;
}

argus_handle* build(std::size_t n_docs) {
    std::mt19937 rng(42);
    const auto vocab = make_vocab(20000);
    argus_handle* h = nullptr;
    argus_open(":memory:", &h);
    for (std::size_t i = 0; i < n_docs; ++i) {
        const auto doc = make_doc(rng, vocab, 400);
        argus_upsert(h, static_cast<int64_t>(i), doc.data(), doc.size());
    }
    return h;
}

}  // namespace

static void BM_Search(benchmark::State& state) {
    argus_handle* h = build(static_cast<std::size_t>(state.range(0)));
    const std::string query = "w3 w17 w50 w120 w400 w900 w1500 w3000 w5000 w9000";
    std::vector<argus_hit> hits(50);
    for (auto _ : state) {
        std::size_t n = 0;
        argus_search_bm25(h, query.data(), query.size(), hits.size(), hits.data(), &n);
        benchmark::DoNotOptimize(n);
    }
    argus_close(h);
}
BENCHMARK(BM_Search)->Arg(3000)->Arg(30000)->Unit(benchmark::kMicrosecond);

static void BM_BuildIndex(benchmark::State& state) {
    for (auto _ : state) {
        argus_handle* h = build(static_cast<std::size_t>(state.range(0)));
        argus_close(h);
    }
}
BENCHMARK(BM_BuildIndex)->Arg(3000)->Unit(benchmark::kMillisecond)->Iterations(2);

BENCHMARK_MAIN();
