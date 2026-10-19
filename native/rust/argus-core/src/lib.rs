//! Safe Rust wrapper over libargus, the local retrieval core.
//!
//! The C library is built by `build.rs` (CMake) and linked statically; the raw
//! declarations are generated from `include/argus.h` by bindgen. This module is the only
//! place that touches them.

#![allow(clippy::missing_safety_doc)]

use std::ffi::{CStr, CString};
use std::ptr;

#[allow(non_camel_case_types, non_upper_case_globals, non_snake_case, dead_code)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("invalid argument")]
    InvalidArgument,
    #[error("path contains an interior NUL byte")]
    InvalidPath,
    #[error("storage error: {0}")]
    Storage(String),
    #[error("out of memory")]
    OutOfMemory,
    #[error("embedding model error: {0}")]
    Model(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// One search result. Higher `score` is better.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub id: i64,
    pub score: f64,
}

/// A handle to an index. Closing happens on drop.
pub struct Index {
    handle: *mut ffi::argus_handle,
}

// SAFETY: libargus serializes every call on one handle behind an internal mutex (see the
// threading note in include/argus.h), so a handle may be moved to and shared across threads.
unsafe impl Send for Index {}
unsafe impl Sync for Index {}

impl Index {
    /// Opens (creating if needed) an index at `path`, or `":memory:"` for a throwaway one.
    pub fn open(path: &str) -> Result<Index> {
        let c_path = CString::new(path).map_err(|_| Error::InvalidPath)?;
        let mut handle: *mut ffi::argus_handle = ptr::null_mut();
        // SAFETY: c_path is a valid NUL-terminated string for the duration of the call and
        // `handle` is a valid out-pointer.
        let status = unsafe { ffi::argus_open(c_path.as_ptr(), &mut handle) };
        match status {
            ffi::argus_status::ARGUS_OK => Ok(Index { handle }),
            // No handle exists on failure, so there is no library error string to read.
            ffi::argus_status::ARGUS_ERR_NOMEM => Err(Error::OutOfMemory),
            ffi::argus_status::ARGUS_ERR_ARG => Err(Error::InvalidArgument),
            ffi::argus_status::ARGUS_ERR_DB => Err(Error::Storage(format!("could not open index at {path}"))),
            ffi::argus_status::ARGUS_ERR_MODEL => Err(Error::Model("unexpected model error opening an index".into())),
        }
    }

    fn check(&self, status: ffi::argus_status) -> Result<()> {
        match status {
            ffi::argus_status::ARGUS_OK => Ok(()),
            ffi::argus_status::ARGUS_ERR_ARG => Err(Error::InvalidArgument),
            ffi::argus_status::ARGUS_ERR_NOMEM => Err(Error::OutOfMemory),
            ffi::argus_status::ARGUS_ERR_MODEL => Err(Error::Model("unexpected model error from an index call".into())),
            ffi::argus_status::ARGUS_ERR_DB => {
                // SAFETY: handle is valid for the life of self; the returned pointer is a
                // NUL-terminated string owned by the library, valid until the next call, and
                // copied out immediately.
                let msg = unsafe { CStr::from_ptr(ffi::argus_last_error(self.handle)) };
                Err(Error::Storage(msg.to_string_lossy().into_owned()))
            }
        }
    }

    /// Like [`Index::open`], but the index also stores `dim`-dimensional vectors and answers dense and
    /// hybrid queries. Reopening an existing file with a different `dim` is an `InvalidArgument` error.
    pub fn open_with_vectors(path: &str, dim: usize) -> Result<Index> {
        let c_path = CString::new(path).map_err(|_| Error::InvalidPath)?;
        let mut handle: *mut ffi::argus_handle = ptr::null_mut();
        // SAFETY: valid NUL-terminated path and out-pointer for the duration of the call.
        let status = unsafe { ffi::argus_open_with_vectors(c_path.as_ptr(), dim, &mut handle) };
        match status {
            ffi::argus_status::ARGUS_OK => Ok(Index { handle }),
            ffi::argus_status::ARGUS_ERR_NOMEM => Err(Error::OutOfMemory),
            ffi::argus_status::ARGUS_ERR_ARG => Err(Error::InvalidArgument),
            _ => Err(Error::Storage(format!("could not open index at {path}"))),
        }
    }

    /// Stores (or replaces) the vector for `id`. Its length must equal the index's dimension.
    pub fn upsert_vector(&self, id: i64, vector: &[f32]) -> Result<()> {
        // SAFETY: the pointer/length pair describes `vector`, which outlives the call.
        self.check(unsafe { ffi::argus_upsert_vector(self.handle, id, vector.as_ptr(), vector.len()) })
    }

    /// Nearest neighbours by cosine similarity; `score` is the cosine. Best first.
    pub fn search_dense(&self, query_vector: &[f32], k: usize) -> Result<Vec<Hit>> {
        let mut buf = vec![ffi::argus_hit { id: 0, score: 0.0 }; k];
        let mut written: usize = 0;
        // SAFETY: `buf` has room for `k` hits; the query pointer/length are valid for the call.
        let status = unsafe {
            ffi::argus_search_dense(self.handle, query_vector.as_ptr(), query_vector.len(), k, buf.as_mut_ptr(), &mut written)
        };
        self.check(status)?;
        buf.truncate(written);
        Ok(buf.into_iter().map(|h| Hit { id: h.id, score: h.score }).collect())
    }

    /// Reciprocal-rank fusion of the BM25 ranking for `query` and the dense ranking for `query_vector`
    /// (each cut at the top 100 candidates). `score` is the fused RRF score. Best first.
    pub fn search_hybrid(&self, query: &str, query_vector: &[f32], k: usize) -> Result<Vec<Hit>> {
        let mut buf = vec![ffi::argus_hit { id: 0, score: 0.0 }; k];
        let mut written: usize = 0;
        // SAFETY: as for search_bm25 and search_dense: all pointers/lengths are valid for the call and
        // `buf` holds `k` hits.
        let status = unsafe {
            ffi::argus_search_hybrid(
                self.handle,
                query.as_ptr().cast(),
                query.len(),
                query_vector.as_ptr(),
                query_vector.len(),
                k,
                buf.as_mut_ptr(),
                &mut written,
            )
        };
        self.check(status)?;
        buf.truncate(written);
        Ok(buf.into_iter().map(|h| Hit { id: h.id, score: h.score }).collect())
    }

    /// Inserts the document, replacing any existing one with the same id.
    pub fn upsert(&self, id: i64, text: &str) -> Result<()> {
        // SAFETY: the pointer/length pair describes `text`, which outlives the call; the library
        // copies what it keeps.
        let status = unsafe { ffi::argus_upsert(self.handle, id, text.as_ptr().cast(), text.len()) };
        self.check(status)
    }

    /// Removes a document. Removing an id that is not present is not an error.
    pub fn remove(&self, id: i64) -> Result<()> {
        // SAFETY: handle is valid for the life of self.
        self.check(unsafe { ffi::argus_remove(self.handle, id) })
    }

    pub fn len(&self) -> Result<usize> {
        let mut n: usize = 0;
        // SAFETY: handle is valid; `n` is a valid out-pointer.
        self.check(unsafe { ffi::argus_count(self.handle, &mut n) })?;
        Ok(n)
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// BM25 keyword search over free text (any punctuation is safe; terms are OR-ed). Returns up
    /// to `k` hits, best first. Documents that match no term are not returned.
    pub fn search_bm25(&self, query: &str, k: usize) -> Result<Vec<Hit>> {
        let mut buf = vec![ffi::argus_hit { id: 0, score: 0.0 }; k];
        let mut written: usize = 0;
        // SAFETY: `buf` has room for `k` hits and is only read up to `written` entries after the
        // call; query's pointer/length are valid for the call; `written` is a valid out-pointer.
        let status = unsafe {
            ffi::argus_search_bm25(self.handle, query.as_ptr().cast(), query.len(), k, buf.as_mut_ptr(), &mut written)
        };
        self.check(status)?;
        buf.truncate(written);
        Ok(buf.into_iter().map(|h| Hit { id: h.id, score: h.score }).collect())
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        // SAFETY: handle came from a successful argus_open and is closed exactly once, here.
        unsafe { ffi::argus_close(self.handle) };
    }
}

/// Embedding vectors for a batch of texts: `len()` rows of `dim()` floats, each L2-normalized, so
/// the dot product of two rows is their cosine similarity.
#[derive(Debug, Clone)]
pub struct Embeddings {
    dim: usize,
    data: Vec<f32>,
    truncated: Vec<bool>,
}

impl Embeddings {
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn len(&self) -> usize {
        self.truncated.len()
    }
    pub fn is_empty(&self) -> bool {
        self.truncated.is_empty()
    }
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
    /// True if text `i` was longer than the model's context and was truncated.
    pub fn was_truncated(&self, i: usize) -> bool {
        self.truncated[i]
    }
}

/// A loaded GGUF embedding model (llama.cpp). Closing happens on drop.
pub struct Embedder {
    handle: *mut ffi::argus_embedder,
    dim: usize,
}

// SAFETY: libargus serializes every call on one embedder behind an internal mutex.
unsafe impl Send for Embedder {}
unsafe impl Sync for Embedder {}

impl Embedder {
    /// Loads the model at `model_path`. `n_threads == 0` picks a default.
    pub fn open(model_path: &str, n_threads: i32) -> Result<Embedder> {
        let c_path = CString::new(model_path).map_err(|_| Error::InvalidPath)?;
        let mut handle: *mut ffi::argus_embedder = ptr::null_mut();
        // SAFETY: valid NUL-terminated path and out-pointer for the duration of the call.
        let status = unsafe { ffi::argus_embedder_open(c_path.as_ptr(), n_threads, &mut handle) };
        match status {
            ffi::argus_status::ARGUS_OK => {}
            ffi::argus_status::ARGUS_ERR_NOMEM => return Err(Error::OutOfMemory),
            ffi::argus_status::ARGUS_ERR_ARG => return Err(Error::InvalidArgument),
            _ => return Err(Error::Model(format!("could not load an embedding model from {model_path}"))),
        }
        let mut dim: usize = 0;
        // SAFETY: handle was just opened successfully; `dim` is a valid out-pointer.
        let status = unsafe { ffi::argus_embedder_dim(handle, &mut dim) };
        debug_assert_eq!(status, ffi::argus_status::ARGUS_OK);
        Ok(Embedder { handle, dim })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Embeds the texts (add the model's task prefix yourself, e.g. "search_query: ").
    pub fn embed<S: AsRef<str>>(&self, texts: &[S]) -> Result<Embeddings> {
        let ptrs: Vec<*const std::os::raw::c_char> = texts.iter().map(|t| t.as_ref().as_ptr().cast()).collect();
        let lens: Vec<usize> = texts.iter().map(|t| t.as_ref().len()).collect();
        let mut data = vec![0.0f32; texts.len() * self.dim];
        let mut trunc = vec![0u8; texts.len()];
        // SAFETY: ptrs/lens describe `texts`, which outlive the call; `data` holds n * dim floats and
        // `trunc` n bytes, matching the sizes passed.
        let status = unsafe {
            ffi::argus_embed_batch(
                self.handle,
                ptrs.as_ptr(),
                lens.as_ptr(),
                texts.len(),
                data.as_mut_ptr(),
                self.dim,
                trunc.as_mut_ptr(),
            )
        };
        match status {
            ffi::argus_status::ARGUS_OK => Ok(Embeddings { dim: self.dim, data, truncated: trunc.into_iter().map(|b| b != 0).collect() }),
            ffi::argus_status::ARGUS_ERR_ARG => Err(Error::InvalidArgument),
            ffi::argus_status::ARGUS_ERR_NOMEM => Err(Error::OutOfMemory),
            _ => {
                // SAFETY: handle valid for the life of self; string copied out immediately.
                let msg = unsafe { CStr::from_ptr(ffi::argus_embedder_last_error(self.handle)) };
                Err(Error::Model(msg.to_string_lossy().into_owned()))
            }
        }
    }
}

impl Drop for Embedder {
    fn drop(&mut self) {
        // SAFETY: handle came from a successful argus_embedder_open and is closed exactly once.
        unsafe { ffi::argus_embedder_close(self.handle) };
    }
}
