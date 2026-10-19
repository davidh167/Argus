/* libargus: local retrieval core, C ABI.
 *
 * Ownership rules:
 *   - Every pointer argument is borrowed for the duration of the call only; the
 *     library copies what it keeps.
 *   - Output buffers (argus_hit arrays) are allocated by the caller and filled by
 *     the library; the library returns nothing it expects the caller to free.
 *   - A argus_handle is created by argus_open and must be released with argus_close.
 * Threading: all calls on one handle are serialized by the library (safe to share
 * across threads); different handles are independent.
 * Errors: every function returns a argus_status and never throws across this boundary.
 * argus_last_error returns a message for the most recent failure on that handle.
 */
#ifndef ARGUS_H
#define ARGUS_H

#include <stddef.h>
#include <stdint.h>

/* The library is built with hidden symbol visibility; only this API is exported. */
#if defined(__GNUC__) || defined(__clang__)
#define ARGUS_API __attribute__((visibility("default")))
#else
#define ARGUS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ARGUS_OK = 0,
    ARGUS_ERR_ARG = 1,   /* NULL/invalid argument */
    ARGUS_ERR_DB = 2,    /* storage engine error; see argus_last_error */
    ARGUS_ERR_NOMEM = 3,
    ARGUS_ERR_MODEL = 4  /* embedding model failed to load or run; see argus_embedder_last_error */
} argus_status;

typedef struct argus_handle argus_handle;

typedef struct {
    int64_t id;    /* the id given to argus_upsert */
    double score;  /* higher is better */
} argus_hit;

/* path: a file path for a persistent index, or ":memory:". */
ARGUS_API argus_status argus_open(const char* path, argus_handle** out);
ARGUS_API void argus_close(argus_handle* h);

/* Like argus_open, but the index also stores dense vectors of dimension dim (> 0) and can answer
 * nearest-neighbour and hybrid queries. Vectors live in the SQLite file (the source of truth);
 * an in-memory HNSW index is rebuilt from them on open. Reopening an existing file with a
 * different dim is an error (ARGUS_ERR_ARG). */
ARGUS_API argus_status argus_open_with_vectors(const char* path, size_t dim, argus_handle** out);

/* Inserts or replaces the document with this id. text is UTF-8, len bytes. */
ARGUS_API argus_status argus_upsert(argus_handle* h, int64_t id, const char* text, size_t len);
ARGUS_API argus_status argus_remove(argus_handle* h, int64_t id);
ARGUS_API argus_status argus_count(argus_handle* h, size_t* out);

/* BM25 keyword search. query is free text (any punctuation is safe); terms are
 * OR-ed. Writes up to capacity hits, best first, into hits and the number written
 * into *n_hits. Documents that match no term are not returned. */
ARGUS_API argus_status argus_search_bm25(argus_handle* h, const char* query, size_t query_len, size_t capacity,
                           argus_hit* hits, size_t* n_hits);

ARGUS_API const char* argus_last_error(const argus_handle* h);

/* ---- Dense and hybrid search (index opened with argus_open_with_vectors) ------------- */

/* Stores (or replaces) the vector for id. dim must equal the index's dimension. The id need
 * not have a text document, and vice versa. argus_remove deletes both. */
ARGUS_API argus_status argus_upsert_vector(argus_handle* h, int64_t id, const float* vec, size_t dim);

/* Nearest neighbours by cosine similarity; score is the cosine in [-1, 1], best first. */
ARGUS_API argus_status argus_search_dense(argus_handle* h, const float* query_vec, size_t dim, size_t capacity,
                                    argus_hit* hits, size_t* n_hits);

/* Reciprocal-rank fusion (k=60) of the BM25 ranking for query and the dense ranking for
 * query_vec, each cut at the top 100 candidates. score is the fused RRF score, best first. */
ARGUS_API argus_status argus_search_hybrid(argus_handle* h, const char* query, size_t query_len, const float* query_vec,
                                     size_t dim, size_t capacity, argus_hit* hits, size_t* n_hits);

/* ---- Embeddings (llama.cpp) ------------------------------------------------------
 * An embedder owns a loaded GGUF embedding model (e.g. nomic-embed-text-v1.5). Vectors
 * are mean-pooled and L2-normalized, so the dot product of two vectors is their cosine.
 * Callers add the model's task prefix themselves ("search_query: ", "search_document: ").
 * Inputs longer than the model's context are truncated (the final special token is kept)
 * and reported through the `truncated` flags. Calls on one embedder are serialized.
 */
typedef struct argus_embedder argus_embedder;

/* n_threads <= 0 picks a default. The model file is read from disk and must be a GGUF
 * embedding model whose pooling type is not "none". */
ARGUS_API argus_status argus_embedder_open(const char* model_path, int32_t n_threads, argus_embedder** out);
ARGUS_API void argus_embedder_close(argus_embedder* e);
ARGUS_API argus_status argus_embedder_dim(const argus_embedder* e, size_t* out);

/* Embeds n texts into out, a caller-allocated row-major n x dim float buffer
 * (dim must equal argus_embedder_dim). truncated, if non-NULL, receives n bytes (1 if that
 * text was truncated). */
ARGUS_API argus_status argus_embed_batch(argus_embedder* e, const char* const* texts, const size_t* lens, size_t n,
                                   float* out, size_t dim, uint8_t* truncated);

ARGUS_API const char* argus_embedder_last_error(const argus_embedder* e);

#ifdef __cplusplus
}
#endif

#endif /* ARGUS_H */
