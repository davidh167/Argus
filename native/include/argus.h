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
 * an in-memory HNSW index is built from them on the first dense or hybrid search (stores and reads back never pay for it). Reopening an existing file with a
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

/* ---- Reading back what was stored ----------------------------------------------------
 * For callers that use the index as a persistent key -> (text, vector) store. found is set to 1 when
 * the entry exists and 0 otherwise (a missing entry is not an error).
 */

/* Stores n (id, text, vector) records in one transaction, each replacing any existing document and
 * vector with that id: all of them are written or none. texts/lens are n UTF-8 strings; vecs is a
 * row-major n x dim float buffer (dim must equal the index's dimension). One commit for the batch is
 * what makes bulk loading fast; calling argus_upsert and argus_upsert_vector per item commits twice
 * per item. n == 0 is a no-op. */
ARGUS_API argus_status argus_upsert_records(argus_handle* h, size_t n, const int64_t* ids, const char* const* texts,
                                      const size_t* lens, const float* vecs, size_t dim);

/* Copies the vector stored for id into out, a caller-allocated buffer of dim floats (dim must equal
 * the index's dimension). Bit-exact with what argus_upsert_vector stored. */
ARGUS_API argus_status argus_get_vector(argus_handle* h, int64_t id, float* out, size_t dim, int* found);

/* Copies the text stored by argus_upsert for id into buf (at most cap bytes, no NUL terminator).
 * *len receives the full length in bytes, which may exceed cap; call again with a larger buffer. */
ARGUS_API argus_status argus_get_text(argus_handle* h, int64_t id, char* buf, size_t cap, size_t* len, int* found);

/* Small string key/value metadata kept in the same file (e.g. which model produced the vectors).
 * Keys are separate from the library's own bookkeeping. argus_meta_get follows the argus_get_text
 * buffer convention. Works on any index, with or without vectors. */
ARGUS_API argus_status argus_meta_set(argus_handle* h, const char* key, const char* value);
ARGUS_API argus_status argus_meta_get(argus_handle* h, const char* key, char* buf, size_t cap, size_t* len, int* found);

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

/* ---- Sparse TF-IDF similarity (same-corpus nearest neighbours) ----------------------
 * An in-memory index over a fixed corpus that reproduces app/services/relevance.py exactly:
 *   tokens  = every maximal run of Python "\w" characters in the full-Unicode lowercase of the text
 *             (Python's str.lower() then re.findall(r"\w+"); implemented with ICU, so it needs no
 *             locale and agrees with Python code point for code point -- see native/README.md for the
 *             one Unicode-version caveat),
 *   tf      = count / len(tokens), idf = log(n_docs / (1 + df)) (zero or negative for common terms;
 *             those terms are kept), weight = tf * idf, similarity = cosine of the weight vectors
 *             (0.0 when either vector is empty or has zero norm). Everything is IEEE double.
 * Sums are Neumaier-compensated like Python 3.12's sum(), so scores agree with Python to ~1e-15.
 * Usage: create, add every document, finalize (computes idf and the vectors), then query. After
 * finalize no more documents can be added and the term strings are released.
 * Ranking: argus_sim_topk returns the highest-similarity documents other than the query document.
 * Ties are broken by insertion order (the order of argus_sim_add calls), which is what Python's stable
 * sorted() does over a dict filled in the same order -- feed documents in the order Python iterates them.
 * Same threading rules as argus_handle: calls on one argus_sim are serialized.
 */
typedef struct argus_sim argus_sim;

ARGUS_API argus_status argus_sim_create(argus_sim** out);
ARGUS_API void argus_sim_free(argus_sim* s);

/* Adds a document. text is UTF-8, len bytes (invalid UTF-8 is ARGUS_ERR_ARG). A duplicate id, or
 * a call after argus_sim_finalize, is ARGUS_ERR_ARG. */
ARGUS_API argus_status argus_sim_add(argus_sim* s, int64_t id, const char* text, size_t len);
ARGUS_API argus_status argus_sim_finalize(argus_sim* s);
ARGUS_API argus_status argus_sim_count(argus_sim* s, size_t* out);

/* Writes up to k hits (best first, the query document excluded) into hits, a caller-allocated
 * array of k entries, and the number written into *n_hits. *found is 0 (and *n_hits 0) when id was
 * never added; that is not an error. Requires argus_sim_finalize. */
ARGUS_API argus_status argus_sim_topk(argus_sim* s, int64_t id, size_t k, argus_hit* hits, size_t* n_hits, int* found);

ARGUS_API const char* argus_sim_last_error(const argus_sim* s);

/* Test hooks for the tokenizer. argus_text_tokenize writes the tokens of text, each followed by
 * '\n' (a newline is never a word character), into buf (at most cap bytes); *len receives the full
 * size, which may exceed cap. argus_unicode_probe reports whether code point cp is a "\w" character
 * and its full lowercase mapping as code points (out, at most cap; *n receives the full count). */
ARGUS_API argus_status argus_text_tokenize(const char* text, size_t len, char* buf, size_t cap, size_t* out_len);
ARGUS_API argus_status argus_unicode_probe(uint32_t cp, int* is_word, uint32_t* lower, size_t cap, size_t* n);

/* "17.0": the Unicode version of the ICU this library is running against, and "78.3": the ICU release. */
ARGUS_API const char* argus_unicode_version(void);
ARGUS_API const char* argus_icu_version(void);

#ifdef __cplusplus
}
#endif

#endif /* ARGUS_H */
