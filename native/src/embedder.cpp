#include <llama.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "argus.h"

namespace {

void quiet_log(ggml_log_level, const char*, void*) {}

void init_backend_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        llama_log_set(quiet_log, nullptr);  // the library is chatty on stderr; errors come back as statuses
        llama_backend_init();
    });
}

struct BatchDeleter {
    void operator()(llama_batch* b) const { llama_batch_free(*b); }
};

}  // namespace

struct argus_embedder {
    std::mutex mu;
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
    int32_t dim = 0;
    int32_t n_ctx = 0;
    uint32_t n_seq_max = 0;
    bool use_encode = false;
    std::string error;

    ~argus_embedder() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }

    argus_status fail(std::string msg) {
        error = std::move(msg);
        return ARGUS_ERR_MODEL;
    }

    // Tokenizes text; truncates to the context size keeping the final special token.
    bool tokenize(const char* text, size_t len, std::vector<llama_token>& tokens, bool& truncated) {
        tokens.resize(len + 8);
        int32_t n = llama_tokenize(vocab, text, static_cast<int32_t>(len), tokens.data(),
                                   static_cast<int32_t>(tokens.size()), /*add_special=*/true, /*parse_special=*/false);
        if (n < 0) {
            tokens.resize(static_cast<size_t>(-n));
            n = llama_tokenize(vocab, text, static_cast<int32_t>(len), tokens.data(),
                               static_cast<int32_t>(tokens.size()), true, false);
        }
        if (n < 0) return false;
        tokens.resize(static_cast<size_t>(n));
        truncated = false;
        if (n > n_ctx) {
            const llama_token last = tokens.back();
            tokens.resize(static_cast<size_t>(n_ctx));
            tokens.back() = last;
            truncated = true;
        }
        return true;
    }

    // Runs one forward pass over a group of sequences (total tokens <= n_ctx) and writes one
    // normalized vector per sequence to out (row-major, stride dim).
    argus_status run_group(const std::vector<std::vector<llama_token>>& seqs, float* out) {
        int32_t total = 0;
        for (const auto& s : seqs) total += static_cast<int32_t>(s.size());
        llama_batch raw = llama_batch_init(total, 0, 1);
        std::unique_ptr<llama_batch, BatchDeleter> batch(&raw);
        int32_t n = 0;
        for (std::size_t s = 0; s < seqs.size(); ++s) {
            for (std::size_t i = 0; i < seqs[s].size(); ++i, ++n) {
                raw.token[n] = seqs[s][i];
                raw.pos[n] = static_cast<llama_pos>(i);
                raw.n_seq_id[n] = 1;
                raw.seq_id[n][0] = static_cast<llama_seq_id>(s);
                raw.logits[n] = true;
            }
        }
        raw.n_tokens = n;

        if (auto* mem = llama_get_memory(ctx)) llama_memory_clear(mem, true);
        const int32_t rc = use_encode ? llama_encode(ctx, raw) : llama_decode(ctx, raw);
        if (rc != 0) return fail("llama forward pass failed with code " + std::to_string(rc));

        for (std::size_t s = 0; s < seqs.size(); ++s) {
            const float* e = llama_get_embeddings_seq(ctx, static_cast<llama_seq_id>(s));
            if (e == nullptr) return fail("model produced no pooled embedding for a sequence");
            double norm = 0.0;
            for (int32_t k = 0; k < dim; ++k) norm += static_cast<double>(e[k]) * e[k];
            const float inv = norm > 0.0 ? static_cast<float>(1.0 / std::sqrt(norm)) : 0.0f;
            float* row = out + s * static_cast<std::size_t>(dim);
            for (int32_t k = 0; k < dim; ++k) row[k] = e[k] * inv;
        }
        return ARGUS_OK;
    }
};

extern "C" {

argus_status argus_embedder_open(const char* model_path, int32_t n_threads, argus_embedder** out) {
    if (model_path == nullptr || out == nullptr) return ARGUS_ERR_ARG;
    *out = nullptr;
    try {
        init_backend_once();
        auto e = std::make_unique<argus_embedder>();
        llama_model_params mp = llama_model_default_params();
        e->model = llama_model_load_from_file(model_path, mp);
        if (e->model == nullptr) return ARGUS_ERR_MODEL;  // no handle to carry a message; the status says why
        e->vocab = llama_model_get_vocab(e->model);
        e->dim = llama_model_n_embd_out(e->model);
        e->n_ctx = llama_model_n_ctx_train(e->model);
        e->use_encode = llama_model_has_encoder(e->model) && !llama_model_has_decoder(e->model);
        e->n_seq_max = 16;

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = static_cast<uint32_t>(e->n_ctx);
        cp.n_batch = cp.n_ubatch = static_cast<uint32_t>(e->n_ctx);  // an encoder sees a whole group at once
        cp.n_seq_max = e->n_seq_max;
        cp.embeddings = true;
        cp.n_threads = cp.n_threads_batch =
            n_threads > 0 ? n_threads : static_cast<int32_t>(std::max(1u, std::thread::hardware_concurrency() / 2));
        e->ctx = llama_init_from_model(e->model, cp);
        if (e->ctx == nullptr) return ARGUS_ERR_MODEL;
        if (llama_pooling_type(e->ctx) == LLAMA_POOLING_TYPE_NONE) return ARGUS_ERR_MODEL;  // not an embedding model
        *out = e.release();
        return ARGUS_OK;
    } catch (const std::bad_alloc&) {
        return ARGUS_ERR_NOMEM;
    } catch (...) {
        return ARGUS_ERR_MODEL;
    }
}

void argus_embedder_close(argus_embedder* e) { delete e; }

argus_status argus_embedder_dim(const argus_embedder* e, size_t* out) {
    if (e == nullptr || out == nullptr) return ARGUS_ERR_ARG;
    *out = static_cast<size_t>(e->dim);
    return ARGUS_OK;
}

argus_status argus_embed_batch(argus_embedder* e, const char* const* texts, const size_t* lens, size_t n, float* out,
                           size_t dim, uint8_t* truncated) {
    if (e == nullptr || (n > 0 && (texts == nullptr || lens == nullptr || out == nullptr))) return ARGUS_ERR_ARG;
    if (dim != static_cast<size_t>(e->dim)) return ARGUS_ERR_ARG;
    try {
        std::lock_guard<std::mutex> lock(e->mu);
        std::vector<std::vector<llama_token>> group;
        std::size_t group_tokens = 0, group_start = 0;
        auto flush = [&]() -> argus_status {
            if (group.empty()) return ARGUS_OK;
            const argus_status st = e->run_group(group, out + group_start * dim);
            group_start += group.size();
            group.clear();
            group_tokens = 0;
            return st;
        };
        for (std::size_t i = 0; i < n; ++i) {
            std::vector<llama_token> tokens;
            bool was_truncated = false;
            if (!e->tokenize(texts[i], lens[i], tokens, was_truncated)) return e->fail("tokenization failed");
            if (truncated) truncated[i] = was_truncated ? 1 : 0;
            if (!group.empty() && (group.size() >= e->n_seq_max || group_tokens + tokens.size() > static_cast<std::size_t>(e->n_ctx))) {
                if (const auto st = flush(); st != ARGUS_OK) return st;
            }
            group_tokens += tokens.size();
            group.push_back(std::move(tokens));
        }
        return flush();
    } catch (const std::bad_alloc&) {
        return ARGUS_ERR_NOMEM;
    }
}

const char* argus_embedder_last_error(const argus_embedder* e) { return e ? e->error.c_str() : ""; }

}  // extern "C"
