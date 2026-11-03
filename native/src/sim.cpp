// Sparse TF-IDF similarity: a C++ reproduction of app/services/relevance.py (tokenize, tf, idf,
// tfidf_vector, cosine_similarity, and the stable top-k of find_similar_postings). See argus.h for the
// contract. Compiled with -ffp-contract=off so no multiply-add is fused: the arithmetic must be the
// plain IEEE double operations Python performs.
#include "argus.h"

#include <unicode/uchar.h>
#include <unicode/ustring.h>
#include <unicode/utf16.h>
#include <unicode/utf8.h>
#include <unicode/uversion.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

// ---- tokenizer: re.findall(r"\w+", text.lower()) ----------------------------------------

// Python's \w for str patterns is str.isalnum() or '_'. isalnum is isalpha (general category L*) or
// isdecimal or isdigit or isnumeric; every decimal and digit also has a numeric value, so "has a
// numeric value" covers the last three. (ICU's own u_isalnum/u_isalpha are different sets: Alphabetic
// includes some marks, and u_isalnum excludes numeric No/Nl.)
bool is_word_char(UChar32 c) {
    if (c == '_') return true;
    switch (u_charType(c)) {
        case U_UPPERCASE_LETTER:
        case U_LOWERCASE_LETTER:
        case U_TITLECASE_LETTER:
        case U_MODIFIER_LETTER:
        case U_OTHER_LETTER:
            return true;
        default:
            return u_getNumericValue(c) != U_NO_NUMERIC_VALUE;
    }
}

// UTF-8 -> full Unicode lowercase (root locale, so SpecialCasing such as U+0130 and the context rule for
// final sigma apply, as in Python's str.lower()) -> maximal runs of word characters, each handed to emit as
// UTF-8. Lowercasing the whole string first matters: its result can contain combining marks that are not
// word characters and so split tokens. Scratch buffers are members so repeated calls do not reallocate.
class Tokenizer {
public:
    // Returns false and sets error() on invalid UTF-8 or an ICU failure.
    template <class Emit>
    bool run(const char* text, std::size_t len, Emit&& emit) {
        if (len > static_cast<std::size_t>(INT32_MAX) / 4) return fail("text too large");
        if (is_ascii(text, len)) return run_ascii(text, len, emit);
        utf16_.resize(len + 1);
        UErrorCode ec = U_ZERO_ERROR;
        int32_t n16 = 0;
        u_strFromUTF8(utf16_.data(), static_cast<int32_t>(utf16_.size()), &n16, text, static_cast<int32_t>(len), &ec);
        if (U_FAILURE(ec)) return fail("text is not valid UTF-8");

        lower_.resize(static_cast<std::size_t>(n16) + 16);
        ec = U_ZERO_ERROR;
        int32_t nl = u_strToLower(lower_.data(), static_cast<int32_t>(lower_.size()), utf16_.data(), n16, "", &ec);
        if (ec == U_BUFFER_OVERFLOW_ERROR) {
            lower_.resize(static_cast<std::size_t>(nl));
            ec = U_ZERO_ERROR;
            nl = u_strToLower(lower_.data(), static_cast<int32_t>(lower_.size()), utf16_.data(), n16, "", &ec);
        }
        if (U_FAILURE(ec)) return fail(std::string("u_strToLower failed: ") + u_errorName(ec));

        token_.clear();
        for (int32_t i = 0; i < nl;) {
            UChar32 c;
            U16_NEXT(lower_.data(), i, nl, c);
            if (is_word_char(c)) {
                append_utf8(c);
            } else if (!token_.empty()) {
                emit(std::string_view(token_));
                token_.clear();
            }
        }
        if (!token_.empty()) emit(std::string_view(token_));
        return true;
    }

    const std::string& error() const { return error_; }

private:
    static bool is_ascii(const char* text, std::size_t len) {
        for (std::size_t i = 0; i < len; ++i) {
            if (static_cast<unsigned char>(text[i]) >= 0x80) return false;
        }
        return true;
    }

    // Pure-ASCII text, the common case: lowercase is A-Z -> a-z with no context rule, and the word characters
    // are [0-9A-Za-z_] (no other ASCII character is a letter or has a numeric value). Same result as the ICU
    // path below, without the UTF-16 round trip; tests/test_native_similarity.py compares both with Python.
    template <class Emit>
    bool run_ascii(const char* text, std::size_t len, Emit& emit) {
        token_.clear();
        for (std::size_t i = 0; i < len; ++i) {
            const char c = text[i];
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_') {
                token_.push_back(c);
            } else if (c >= 'A' && c <= 'Z') {
                token_.push_back(static_cast<char>(c + ('a' - 'A')));
            } else if (!token_.empty()) {
                emit(std::string_view(token_));
                token_.clear();
            }
        }
        if (!token_.empty()) emit(std::string_view(token_));
        return true;
    }

    bool fail(std::string msg) {
        error_ = std::move(msg);
        return false;
    }
    void append_utf8(UChar32 c) {
        uint8_t buf[4];
        int32_t n = 0;
        U8_APPEND_UNSAFE(buf, n, c);
        token_.append(reinterpret_cast<const char*>(buf), static_cast<std::size_t>(n));
    }

    std::vector<UChar> utf16_, lower_;
    std::string token_, error_;
};

// ---- Python 3.12's sum() of floats: Neumaier-compensated ------------------------------------
// Reproducing it makes norms bit-identical to Python's and dot products agree to the last bit or so, whatever
// order the terms are visited in.
struct Sum {
    double f = 0.0, c = 0.0;
    void add(double x) {
        const double t = f + x;
        if (std::fabs(f) >= std::fabs(x)) {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    double value() const { return (c != 0.0 && std::isfinite(c)) ? f + c : f; }
};

struct StringHash {
    using is_transparent = void;
    std::size_t operator()(std::string_view s) const { return std::hash<std::string_view>{}(s); }
};

}  // namespace

struct argus_sim {
    std::mutex mu;
    std::string error;
    Tokenizer tokenizer;
    bool finalized = false;

    // Build phase: vocabulary and each document's (term, count) pairs in first-occurrence order -- the order of
    // a Python Counter, and so of the dict whose values Python sums for the norm.
    std::unordered_map<std::string, uint32_t, StringHash, std::equal_to<>> vocab;
    std::vector<uint32_t> term_scratch;  // counts of the document being added, indexed by term id
    std::vector<uint32_t> order_scratch;
    std::vector<int64_t> ids;
    std::unordered_map<int64_t, uint32_t> position;
    std::vector<uint64_t> row_start{0};
    std::vector<uint32_t> terms;
    std::vector<uint32_t> counts;  // dropped by finalize
    std::vector<uint32_t> doc_len;

    // Query phase.
    std::vector<double> weights;  // parallel to terms
    std::vector<double> norms;    // sqrt of the sum of squared weights, per document
    std::vector<double> query_scratch;  // dense weights of the query document, zero elsewhere

    argus_status fail(argus_status st, std::string msg) {
        error = std::move(msg);
        return st;
    }
};

extern "C" {

argus_status argus_sim_create(argus_sim** out) {
    if (out == nullptr) return ARGUS_ERR_ARG;
    *out = nullptr;
    try {
        *out = new argus_sim();
        return ARGUS_OK;
    } catch (const std::bad_alloc&) {
        return ARGUS_ERR_NOMEM;
    }
}

void argus_sim_free(argus_sim* s) { delete s; }

argus_status argus_sim_add(argus_sim* s, int64_t id, const char* text, size_t len) {
    if (s == nullptr || (text == nullptr && len > 0)) return ARGUS_ERR_ARG;
    try {
        std::lock_guard<std::mutex> lock(s->mu);
        if (s->finalized) return s->fail(ARGUS_ERR_ARG, "argus_sim_add after argus_sim_finalize");
        if (s->position.count(id) != 0) return s->fail(ARGUS_ERR_ARG, "duplicate document id " + std::to_string(id));

        uint32_t total = 0;
        s->order_scratch.clear();
        const bool ok = s->tokenizer.run(text ? text : "", len, [&](std::string_view token) {
            auto it = s->vocab.find(token);
            if (it == s->vocab.end()) {
                it = s->vocab.emplace(std::string(token), static_cast<uint32_t>(s->vocab.size())).first;
            }
            const uint32_t term = it->second;
            if (term >= s->term_scratch.size()) s->term_scratch.resize(std::max<std::size_t>(term + 1, s->term_scratch.size() * 2), 0);
            if (s->term_scratch[term]++ == 0) s->order_scratch.push_back(term);
            ++total;
        });
        if (!ok) {
            for (const uint32_t term : s->order_scratch) s->term_scratch[term] = 0;
            return s->fail(ARGUS_ERR_ARG, s->tokenizer.error());
        }
        for (const uint32_t term : s->order_scratch) {
            s->terms.push_back(term);
            s->counts.push_back(s->term_scratch[term]);
            s->term_scratch[term] = 0;
        }
        s->position.emplace(id, static_cast<uint32_t>(s->ids.size()));
        s->ids.push_back(id);
        s->doc_len.push_back(total);
        s->row_start.push_back(s->terms.size());
        return ARGUS_OK;
    } catch (const std::bad_alloc&) {
        return ARGUS_ERR_NOMEM;
    }
}

argus_status argus_sim_finalize(argus_sim* s) {
    if (s == nullptr) return ARGUS_ERR_ARG;
    try {
        std::lock_guard<std::mutex> lock(s->mu);
        if (s->finalized) return s->fail(ARGUS_ERR_ARG, "argus_sim_finalize called twice");
        const std::size_t n_docs = s->ids.size();
        const std::size_t n_terms = s->vocab.size();

        std::vector<uint32_t> df(n_terms, 0);
        for (const uint32_t term : s->terms) ++df[term];  // each document lists a term once
        // Python: n_docs=len(rows) or 1; log(n_docs / (1 + df)), true division of ints.
        const double n = static_cast<double>(n_docs == 0 ? 1 : n_docs);
        std::vector<double> idf(n_terms);
        for (std::size_t t = 0; t < n_terms; ++t) idf[t] = std::log(n / static_cast<double>(1 + df[t]));

        s->weights.resize(s->terms.size());
        s->norms.assign(n_docs, 0.0);
        for (std::size_t d = 0; d < n_docs; ++d) {
            const double total = static_cast<double>(s->doc_len[d]);
            Sum sq;
            for (uint64_t j = s->row_start[d]; j < s->row_start[d + 1]; ++j) {
                const double tf = static_cast<double>(s->counts[j]) / total;
                const double w = tf * idf[s->terms[j]];
                s->weights[j] = w;
                sq.add(w * w);
            }
            s->norms[d] = std::sqrt(sq.value());
        }
        s->query_scratch.assign(n_terms, 0.0);

        // Only ids and the vectors are needed from here on.
        s->vocab = {};
        s->counts = {};
        s->term_scratch = {};
        s->order_scratch = {};
        s->terms.shrink_to_fit();
        s->weights.shrink_to_fit();
        s->finalized = true;
        return ARGUS_OK;
    } catch (const std::bad_alloc&) {
        return ARGUS_ERR_NOMEM;
    }
}

argus_status argus_sim_count(argus_sim* s, size_t* out) {
    if (s == nullptr || out == nullptr) return ARGUS_ERR_ARG;
    std::lock_guard<std::mutex> lock(s->mu);
    *out = s->ids.size();
    return ARGUS_OK;
}

argus_status argus_sim_topk(argus_sim* s, int64_t id, size_t k, argus_hit* hits, size_t* n_hits, int* found) {
    if (s == nullptr || n_hits == nullptr || found == nullptr || (hits == nullptr && k > 0)) return ARGUS_ERR_ARG;
    *n_hits = 0;
    *found = 0;
    try {
        std::lock_guard<std::mutex> lock(s->mu);
        if (!s->finalized) return s->fail(ARGUS_ERR_ARG, "argus_sim_topk before argus_sim_finalize");
        const auto it = s->position.find(id);
        if (it == s->position.end()) return ARGUS_OK;
        *found = 1;
        if (k == 0) return ARGUS_OK;

        const std::size_t n_docs = s->ids.size();
        const uint32_t target = it->second;
        std::vector<double> sims(n_docs, 0.0);
        const double norm_t = s->norms[target];
        if (norm_t != 0.0) {  // an empty vector has norm 0 and scores 0.0 against everything
            for (uint64_t j = s->row_start[target]; j < s->row_start[target + 1]; ++j) {
                s->query_scratch[s->terms[j]] = s->weights[j];
            }
            for (std::size_t d = 0; d < n_docs; ++d) {
                if (d == target || s->norms[d] == 0.0) continue;
                Sum dot;
                for (uint64_t j = s->row_start[d]; j < s->row_start[d + 1]; ++j) {
                    const double q = s->query_scratch[s->terms[j]];
                    if (q != 0.0) dot.add(q * s->weights[j]);  // a zero query weight adds exactly 0
                }
                sims[d] = dot.value() / (norm_t * s->norms[d]);
            }
            for (uint64_t j = s->row_start[target]; j < s->row_start[target + 1]; ++j) s->query_scratch[s->terms[j]] = 0.0;
        }

        // Highest similarity first, ties by insertion order: a total order, so partial_sort is deterministic.
        std::vector<uint32_t> order;
        order.reserve(n_docs);
        for (std::size_t d = 0; d < n_docs; ++d) {
            if (d != target) order.push_back(static_cast<uint32_t>(d));
        }
        const std::size_t take = std::min(k, order.size());
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(take), order.end(),
                          [&](uint32_t a, uint32_t b) { return sims[a] != sims[b] ? sims[a] > sims[b] : a < b; });
        for (std::size_t i = 0; i < take; ++i) hits[i] = {s->ids[order[i]], sims[order[i]]};
        *n_hits = take;
        return ARGUS_OK;
    } catch (const std::bad_alloc&) {
        return ARGUS_ERR_NOMEM;
    }
}

const char* argus_sim_last_error(const argus_sim* s) { return s ? s->error.c_str() : ""; }

argus_status argus_text_tokenize(const char* text, size_t len, char* buf, size_t cap, size_t* out_len) {
    if ((text == nullptr && len > 0) || out_len == nullptr || (buf == nullptr && cap > 0)) return ARGUS_ERR_ARG;
    try {
        Tokenizer tokenizer;
        std::string joined;
        if (!tokenizer.run(text ? text : "", len, [&](std::string_view token) {
                joined.append(token);
                joined.push_back('\n');
            })) {
            return ARGUS_ERR_ARG;
        }
        *out_len = joined.size();
        if (!joined.empty()) std::memcpy(buf, joined.data(), std::min(cap, joined.size()));
        return ARGUS_OK;
    } catch (const std::bad_alloc&) {
        return ARGUS_ERR_NOMEM;
    }
}

argus_status argus_unicode_probe(uint32_t cp, int* is_word, uint32_t* lower, size_t cap, size_t* n) {
    if (cp > 0x10FFFF || is_word == nullptr || n == nullptr || (lower == nullptr && cap > 0)) return ARGUS_ERR_ARG;
    try {
        *is_word = is_word_char(static_cast<UChar32>(cp)) ? 1 : 0;
        UChar src[2];
        int32_t src_len = 0;
        U16_APPEND_UNSAFE(src, src_len, static_cast<UChar32>(cp));
        std::vector<UChar> dst(8);
        UErrorCode ec = U_ZERO_ERROR;
        int32_t dst_len = u_strToLower(dst.data(), static_cast<int32_t>(dst.size()), src, src_len, "", &ec);
        if (ec == U_BUFFER_OVERFLOW_ERROR) {
            dst.resize(static_cast<std::size_t>(dst_len));
            ec = U_ZERO_ERROR;
            dst_len = u_strToLower(dst.data(), static_cast<int32_t>(dst.size()), src, src_len, "", &ec);
        }
        if (U_FAILURE(ec)) return ARGUS_ERR_ARG;
        std::size_t count = 0;
        for (int32_t i = 0; i < dst_len;) {
            UChar32 c;
            U16_NEXT(dst.data(), i, dst_len, c);
            if (count < cap) lower[count] = static_cast<uint32_t>(c);
            ++count;
        }
        *n = count;
        return ARGUS_OK;
    } catch (const std::bad_alloc&) {
        return ARGUS_ERR_NOMEM;
    }
}

const char* argus_unicode_version(void) {
    static const std::string version = [] {
        UVersionInfo info;
        char buf[U_MAX_VERSION_STRING_LENGTH];
        u_getUnicodeVersion(info);
        u_versionToString(info, buf);
        return std::string(buf);
    }();
    return version.c_str();
}

const char* argus_icu_version(void) {
    static const std::string version = [] {
        UVersionInfo info;
        char buf[U_MAX_VERSION_STRING_LENGTH];
        u_getVersion(info);
        u_versionToString(info, buf);
        return std::string(buf);
    }();
    return version.c_str();
}

}  // extern "C"
