#include "argus.h"

#include <sqlite3.h>
#include <usearch/index_dense.hpp>

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using unum::usearch::index_dense_t;
using unum::usearch::index_limits_t;
using unum::usearch::metric_kind_t;
using unum::usearch::metric_punned_t;
using unum::usearch::scalar_kind_t;

namespace {

constexpr std::size_t kHybridCandidates = 100;  // per ranking, before fusion
constexpr double kRrfK = 60.0;                  // reciprocal-rank-fusion constant

// Both mixed into a hybrid result as 1 / (k + rank), rank starting at 1.
double rrf(std::size_t rank0) { return 1.0 / (kRrfK + static_cast<double>(rank0) + 1.0); }


struct StmtDeleter {
    void operator()(sqlite3_stmt* s) const { sqlite3_finalize(s); }
};
using Stmt = std::unique_ptr<sqlite3_stmt, StmtDeleter>;

struct DbDeleter {
    void operator()(sqlite3* db) const { sqlite3_close_v2(db); }
};
using Db = std::unique_ptr<sqlite3, DbDeleter>;

// FTS5 treats a bare query as a mini-language (AND/OR/NOT, quotes, parentheses, '-',
// '*', ':'). A free-text query must never be parsed that way, so every
// whitespace-separated token is wrapped in double quotes (an FTS5 string, tokenized
// by the index's own Unicode tokenizer) and the terms are OR-ed.
std::string to_fts5_query(std::string_view query) {
    std::string out;
    std::size_t i = 0;
    const auto is_space = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
    while (i < query.size()) {
        while (i < query.size() && is_space(query[i])) ++i;
        const std::size_t start = i;
        while (i < query.size() && !is_space(query[i])) ++i;
        if (i == start) break;
        if (!out.empty()) out += " OR ";
        out.push_back('"');
        for (std::size_t k = start; k < i; ++k) {
            if (query[k] == '"') out.push_back('"');  // double any embedded quote
            out.push_back(query[k]);
        }
        out.push_back('"');
    }
    return out;
}

}  // namespace

namespace {

// Resets a prepared statement when it goes out of scope, so an early return cannot
// leave it holding a read lock.
struct StmtReset {
    sqlite3_stmt* s;
    explicit StmtReset(sqlite3_stmt* stmt) : s(stmt) {}
    ~StmtReset() {
        sqlite3_reset(s);
        sqlite3_clear_bindings(s);
    }
};

}  // namespace

struct argus_handle {
    std::mutex mu;
    Db db;
    Stmt del, ins, count, search, vec_del, vec_ins, vec_all;
    std::string error;
    std::size_t dim = 0;                   // 0: no vector index
    std::unique_ptr<index_dense_t> vindex;  // in-memory HNSW, rebuilt from the vecs table on open

    bool fail(int rc) {
        error = db ? sqlite3_errmsg(db.get()) : sqlite3_errstr(rc);
        return false;
    }

    bool prepare(Stmt& out, const char* sql) {
        sqlite3_stmt* raw = nullptr;
        const int rc = sqlite3_prepare_v2(db.get(), sql, -1, &raw, nullptr);
        out.reset(raw);
        return rc == SQLITE_OK || fail(rc);
    }

    bool exec(const char* sql) {
        char* msg = nullptr;
        const int rc = sqlite3_exec(db.get(), sql, nullptr, nullptr, &msg);
        if (rc != SQLITE_OK) {
            error = msg ? msg : sqlite3_errstr(rc);
            sqlite3_free(msg);
            return false;
        }
        return true;
    }

    // ---- vector index (only when dim > 0) ----

    void ensure_capacity() {
        if (vindex->size() >= vindex->capacity()) {
            vindex->reserve(index_limits_t(std::max<std::size_t>(1024, vindex->capacity() * 2), 1));
        }
    }

    // Replaces the vector for id in the in-memory index.
    argus_status index_put(int64_t id, const float* vec) {
        const auto key = static_cast<unum::usearch::default_key_t>(id);
        if (vindex->contains(key)) vindex->remove(key);
        ensure_capacity();
        const auto added = vindex->add(key, vec);
        if (!added) {
            error = added.error.what();
            return ARGUS_ERR_DB;
        }
        return ARGUS_OK;
    }

    // Creates an empty HNSW index and refills it from the vecs table.
    argus_status rebuild_vector_index() {
        metric_punned_t metric(dim, metric_kind_t::cos_k, scalar_kind_t::f32_k);
        unum::usearch::index_dense_config_t config;
        config.expansion_search = 256;  // keep recall high for the top-100 candidate lists hybrid search needs
        auto made = index_dense_t::make(metric, config, unum::usearch::default_free_value<unum::usearch::default_key_t>(),
                                        index_limits_t(1024, 1));
        if (!made) {
            error = made.error.what();
            return ARGUS_ERR_DB;
        }
        vindex = std::make_unique<index_dense_t>(std::move(made.index));
        StmtReset reset(vec_all.get());
        int rc;
        while ((rc = sqlite3_step(vec_all.get())) == SQLITE_ROW) {
            const auto* blob = static_cast<const float*>(sqlite3_column_blob(vec_all.get(), 1));
            if (static_cast<std::size_t>(sqlite3_column_bytes(vec_all.get(), 1)) != dim * sizeof(float)) {
                error = "stored vector has the wrong size";
                return ARGUS_ERR_DB;
            }
            if (const auto st = index_put(sqlite3_column_int64(vec_all.get(), 0), blob); st != ARGUS_OK) return st;
        }
        if (rc != SQLITE_DONE) {
            fail(rc);
            return ARGUS_ERR_DB;
        }
        return ARGUS_OK;
    }
};

static argus_status open_impl(const char* path, std::size_t dim, argus_handle** out) {
    if (path == nullptr || out == nullptr) return ARGUS_ERR_ARG;
    *out = nullptr;
    try {
        auto h = std::make_unique<argus_handle>();
        sqlite3* raw = nullptr;
        const int rc = sqlite3_open_v2(path, &raw, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
                                       nullptr);
        h->db.reset(raw);
        if (rc != SQLITE_OK) {
            // The handle is unusable and never returned, so there is nobody to read an
            // error message from; callers get the status code.
            return rc == SQLITE_NOMEM ? ARGUS_ERR_NOMEM : ARGUS_ERR_DB;
        }
        if (!h->exec("PRAGMA journal_mode=WAL;") ||
            !h->exec("CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(text, tokenize='unicode61 remove_diacritics 2');") ||
            !h->exec("CREATE TABLE IF NOT EXISTS vecs(id INTEGER PRIMARY KEY, v BLOB NOT NULL);") ||
            !h->exec("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);") ||
            !h->prepare(h->del, "DELETE FROM docs WHERE rowid = ?1;") ||
            !h->prepare(h->ins, "INSERT INTO docs(rowid, text) VALUES(?1, ?2);") ||
            !h->prepare(h->count, "SELECT count(*) FROM docs;") ||
            !h->prepare(h->vec_del, "DELETE FROM vecs WHERE id = ?1;") ||
            !h->prepare(h->vec_ins, "INSERT OR REPLACE INTO vecs(id, v) VALUES(?1, ?2);") ||
            !h->prepare(h->vec_all, "SELECT id, v FROM vecs;") ||
            !h->prepare(h->search, "SELECT rowid, -bm25(docs) FROM docs WHERE docs MATCH ?1 ORDER BY bm25(docs) LIMIT ?2;")) {
            return ARGUS_ERR_DB;
        }
        if (dim > 0) {
            // The dimension is fixed per file: reopening with another one would misread every vector.
            Stmt get;
            if (!h->prepare(get, "SELECT v FROM meta WHERE k = 'dim';")) return ARGUS_ERR_DB;
            if (sqlite3_step(get.get()) == SQLITE_ROW) {
                if (static_cast<std::size_t>(sqlite3_column_int64(get.get(), 0)) != dim) return ARGUS_ERR_ARG;
            } else if (!h->exec(("INSERT INTO meta(k, v) VALUES('dim', '" + std::to_string(dim) + "');").c_str())) {
                return ARGUS_ERR_DB;
            }
            h->dim = dim;
            if (const auto st = h->rebuild_vector_index(); st != ARGUS_OK) return st;
        }
        *out = h.release();
        return ARGUS_OK;
    } catch (const std::bad_alloc&) {
        return ARGUS_ERR_NOMEM;
    } catch (...) {
        return ARGUS_ERR_DB;
    }
}

extern "C" {

argus_status argus_open(const char* path, argus_handle** out) { return open_impl(path, 0, out); }

argus_status argus_open_with_vectors(const char* path, size_t dim, argus_handle** out) {
    if (dim == 0) return ARGUS_ERR_ARG;
    return open_impl(path, dim, out);
}

void argus_close(argus_handle* h) { delete h; }

argus_status argus_upsert(argus_handle* h, int64_t id, const char* text, size_t len) {
    if (h == nullptr || (text == nullptr && len > 0)) return ARGUS_ERR_ARG;
    try {
        std::lock_guard<std::mutex> lock(h->mu);
        if (!h->exec("BEGIN;")) return ARGUS_ERR_DB;
        bool ok = true;
        {
            StmtReset r1(h->del.get());
            sqlite3_bind_int64(h->del.get(), 1, id);
            ok = sqlite3_step(h->del.get()) == SQLITE_DONE || h->fail(sqlite3_errcode(h->db.get()));
        }
        if (ok) {
            StmtReset r2(h->ins.get());
            sqlite3_bind_int64(h->ins.get(), 1, id);
            sqlite3_bind_text(h->ins.get(), 2, text ? text : "", static_cast<int>(len), SQLITE_STATIC);
            ok = sqlite3_step(h->ins.get()) == SQLITE_DONE || h->fail(sqlite3_errcode(h->db.get()));
        }
        if (!ok) {
            h->exec("ROLLBACK;");
            return ARGUS_ERR_DB;
        }
        return h->exec("COMMIT;") ? ARGUS_OK : ARGUS_ERR_DB;
    } catch (const std::bad_alloc&) {
        return ARGUS_ERR_NOMEM;
    }
}

argus_status argus_remove(argus_handle* h, int64_t id) {
    if (h == nullptr) return ARGUS_ERR_ARG;
    try {
        std::lock_guard<std::mutex> lock(h->mu);
        for (sqlite3_stmt* stmt : {h->del.get(), h->vec_del.get()}) {
            StmtReset r(stmt);
            sqlite3_bind_int64(stmt, 1, id);
            if (sqlite3_step(stmt) != SQLITE_DONE) {
                h->fail(sqlite3_errcode(h->db.get()));
                return ARGUS_ERR_DB;
            }
        }
        if (h->vindex) {
            const auto key = static_cast<unum::usearch::default_key_t>(id);
            if (h->vindex->contains(key)) h->vindex->remove(key);
        }
        return ARGUS_OK;
    } catch (const std::bad_alloc&) {
        return ARGUS_ERR_NOMEM;
    }
}

argus_status argus_count(argus_handle* h, size_t* out) {
    if (h == nullptr || out == nullptr) return ARGUS_ERR_ARG;
    try {
        std::lock_guard<std::mutex> lock(h->mu);
        StmtReset r(h->count.get());
        if (sqlite3_step(h->count.get()) != SQLITE_ROW) {
            h->fail(sqlite3_errcode(h->db.get()));
            return ARGUS_ERR_DB;
        }
        *out = static_cast<size_t>(sqlite3_column_int64(h->count.get(), 0));
        return ARGUS_OK;
    } catch (const std::bad_alloc&) {
        return ARGUS_ERR_NOMEM;
    }
}

}  // extern "C"

namespace {

// BM25 ranking, best first, at most capacity entries. Caller holds h->mu.
argus_status bm25_locked(argus_handle* h, const std::string& fts, std::size_t capacity, std::vector<argus_hit>& out) {
    out.clear();
    if (fts.empty() || capacity == 0) return ARGUS_OK;
    StmtReset r(h->search.get());
    sqlite3_bind_text(h->search.get(), 1, fts.c_str(), static_cast<int>(fts.size()), SQLITE_STATIC);
    sqlite3_bind_int64(h->search.get(), 2, static_cast<sqlite3_int64>(capacity));
    int rc;
    while ((rc = sqlite3_step(h->search.get())) == SQLITE_ROW) {
        out.push_back({sqlite3_column_int64(h->search.get(), 0), sqlite3_column_double(h->search.get(), 1)});
    }
    if (rc != SQLITE_DONE) {
        h->fail(rc);
        return ARGUS_ERR_DB;
    }
    return ARGUS_OK;
}

// Cosine-similarity ranking, best first. Caller holds h->mu and has checked h->vindex.
argus_status dense_locked(argus_handle* h, const float* q, std::size_t capacity, std::vector<argus_hit>& out) {
    out.clear();
    if (capacity == 0 || h->vindex->size() == 0) return ARGUS_OK;
    const auto found = h->vindex->search(q, capacity);
    if (!found) {
        h->error = found.error.what();
        return ARGUS_ERR_DB;
    }
    for (std::size_t i = 0; i < found.size(); ++i) {
        out.push_back({static_cast<int64_t>(found[i].member.key), 1.0 - static_cast<double>(found[i].distance)});
    }
    return ARGUS_OK;
}

argus_status copy_out(const std::vector<argus_hit>& hits, std::size_t capacity, argus_hit* out, size_t* n) {
    *n = std::min(capacity, hits.size());
    std::copy_n(hits.begin(), *n, out);
    return ARGUS_OK;
}

}  // namespace

extern "C" {

argus_status argus_search_bm25(argus_handle* h, const char* query, size_t query_len, size_t capacity, argus_hit* hits,
                           size_t* n_hits) {
    if (h == nullptr || n_hits == nullptr || (query == nullptr && query_len > 0) || (hits == nullptr && capacity > 0)) {
        return ARGUS_ERR_ARG;
    }
    *n_hits = 0;
    try {
        const std::string fts = to_fts5_query(std::string_view(query ? query : "", query_len));
        std::lock_guard<std::mutex> lock(h->mu);
        std::vector<argus_hit> found;
        if (const auto st = bm25_locked(h, fts, capacity, found); st != ARGUS_OK) return st;
        return copy_out(found, capacity, hits, n_hits);
    } catch (const std::bad_alloc&) {
        return ARGUS_ERR_NOMEM;
    }
}

argus_status argus_upsert_vector(argus_handle* h, int64_t id, const float* vec, size_t dim) {
    if (h == nullptr || vec == nullptr || h->dim == 0 || dim != h->dim) return ARGUS_ERR_ARG;
    try {
        std::lock_guard<std::mutex> lock(h->mu);
        if (!h->exec("BEGIN;")) return ARGUS_ERR_DB;
        {
            StmtReset r(h->vec_ins.get());
            sqlite3_bind_int64(h->vec_ins.get(), 1, id);
            sqlite3_bind_blob(h->vec_ins.get(), 2, vec, static_cast<int>(dim * sizeof(float)), SQLITE_STATIC);
            if (sqlite3_step(h->vec_ins.get()) != SQLITE_DONE) {
                h->fail(sqlite3_errcode(h->db.get()));
                h->exec("ROLLBACK;");
                return ARGUS_ERR_DB;
            }
        }
        if (!h->exec("COMMIT;")) return ARGUS_ERR_DB;
        // The file is the source of truth: if the in-memory update fails the next open rebuilds it.
        return h->index_put(id, vec);
    } catch (const std::bad_alloc&) {
        return ARGUS_ERR_NOMEM;
    }
}

argus_status argus_search_dense(argus_handle* h, const float* query_vec, size_t dim, size_t capacity, argus_hit* hits,
                            size_t* n_hits) {
    if (h == nullptr || query_vec == nullptr || n_hits == nullptr || h->dim == 0 || dim != h->dim ||
        (hits == nullptr && capacity > 0)) {
        return ARGUS_ERR_ARG;
    }
    *n_hits = 0;
    try {
        std::lock_guard<std::mutex> lock(h->mu);
        std::vector<argus_hit> found;
        if (const auto st = dense_locked(h, query_vec, capacity, found); st != ARGUS_OK) return st;
        return copy_out(found, capacity, hits, n_hits);
    } catch (const std::bad_alloc&) {
        return ARGUS_ERR_NOMEM;
    }
}

argus_status argus_search_hybrid(argus_handle* h, const char* query, size_t query_len, const float* query_vec, size_t dim,
                             size_t capacity, argus_hit* hits, size_t* n_hits) {
    if (h == nullptr || query_vec == nullptr || n_hits == nullptr || h->dim == 0 || dim != h->dim ||
        (query == nullptr && query_len > 0) || (hits == nullptr && capacity > 0)) {
        return ARGUS_ERR_ARG;
    }
    *n_hits = 0;
    try {
        const std::string fts = to_fts5_query(std::string_view(query ? query : "", query_len));
        std::lock_guard<std::mutex> lock(h->mu);
        std::vector<argus_hit> keyword, dense;
        if (const auto st = bm25_locked(h, fts, kHybridCandidates, keyword); st != ARGUS_OK) return st;
        if (const auto st = dense_locked(h, query_vec, kHybridCandidates, dense); st != ARGUS_OK) return st;

        std::unordered_map<int64_t, double> fused;
        for (std::size_t i = 0; i < keyword.size(); ++i) fused[keyword[i].id] += rrf(i);
        for (std::size_t i = 0; i < dense.size(); ++i) fused[dense[i].id] += rrf(i);
        std::vector<argus_hit> ranked;
        ranked.reserve(fused.size());
        for (const auto& [id, score] : fused) ranked.push_back({id, score});
        std::sort(ranked.begin(), ranked.end(), [](const argus_hit& a, const argus_hit& b) {
            return a.score != b.score ? a.score > b.score : a.id < b.id;  // deterministic ties
        });
        return copy_out(ranked, capacity, hits, n_hits);
    } catch (const std::bad_alloc&) {
        return ARGUS_ERR_NOMEM;
    }
}

const char* argus_last_error(const argus_handle* h) { return h ? h->error.c_str() : ""; }

}  // extern "C"
