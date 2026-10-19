use std::sync::Arc;
use std::thread;

use argus_core::{Error, Index};

fn ids(hits: &[argus_core::Hit]) -> Vec<i64> {
    hits.iter().map(|h| h.id).collect()
}

#[test]
fn ranks_the_more_relevant_document_first() {
    let idx = Index::open(":memory:").unwrap();
    idx.upsert(1, "python machine learning engineer building retrieval systems").unwrap();
    idx.upsert(2, "sales manager for enterprise accounts").unwrap();
    idx.upsert(3, "python python python scripting").unwrap();
    let hits = idx.search_bm25("python retrieval", 10).unwrap();
    assert_eq!(ids(&hits), vec![1, 3]);
    assert!(hits[0].score > hits[1].score);
}

#[test]
fn upsert_replaces_and_remove_deletes() {
    let idx = Index::open(":memory:").unwrap();
    idx.upsert(7, "alpha").unwrap();
    idx.upsert(7, "beta").unwrap();
    assert_eq!(idx.len().unwrap(), 1);
    assert!(idx.search_bm25("alpha", 5).unwrap().is_empty());
    assert_eq!(idx.search_bm25("beta", 5).unwrap().len(), 1);
    idx.remove(7).unwrap();
    assert!(idx.is_empty().unwrap());
    idx.remove(999).unwrap(); // missing id is fine
}

#[test]
fn respects_capacity_including_zero() {
    let idx = Index::open(":memory:").unwrap();
    for i in 0..20 {
        idx.upsert(i, &format!("shared term {i}")).unwrap();
    }
    assert_eq!(idx.search_bm25("shared", 5).unwrap().len(), 5);
    assert_eq!(idx.search_bm25("shared", 100).unwrap().len(), 20);
    assert!(idx.search_bm25("shared", 0).unwrap().is_empty());
}

#[test]
fn query_syntax_characters_never_error() {
    let idx = Index::open(":memory:").unwrap();
    idx.upsert(1, "c++ and rust systems programming").unwrap();
    for q in ["c++", "\"unbalanced", "AND OR NOT", "(parens", "col:val", "foo*", "-minus", "&", "", "  ", "NEAR(x y)"] {
        idx.search_bm25(q, 10).unwrap_or_else(|e| panic!("query {q:?} failed: {e}"));
    }
}

#[test]
fn unicode_is_case_and_accent_insensitive() {
    let idx = Index::open(":memory:").unwrap();
    idx.upsert(1, "R\u{c9}SUM\u{c9} parsing for caf\u{e9} owners").unwrap();
    idx.upsert(2, "company\u{2019}s AI\u{2014}driven platform").unwrap();
    assert_eq!(idx.search_bm25("resume", 5).unwrap().len(), 1);
    assert_eq!(idx.search_bm25("CAFE", 5).unwrap().len(), 1);
    assert_eq!(idx.search_bm25("driven", 5).unwrap().len(), 1);
}

#[test]
fn open_failures_are_errors_not_panics() {
    assert!(matches!(Index::open("/nonexistent-dir/x/y.db"), Err(Error::Storage(_))));
    assert_eq!(Index::open("bad\0path").err(), Some(Error::InvalidPath));
}

#[test]
fn a_persistent_index_survives_reopen() {
    let path = std::env::temp_dir().join(format!("argus_rust_test_{}.db", std::process::id()));
    let p = path.to_str().unwrap();
    {
        let idx = Index::open(p).unwrap();
        idx.upsert(1, "persistent document about retrieval").unwrap();
    }
    {
        let idx = Index::open(p).unwrap();
        assert_eq!(idx.search_bm25("retrieval", 5).unwrap().len(), 1);
    }
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{p}{suffix}"));
    }
}

#[test]
fn a_shared_index_is_safe_across_threads() {
    let idx = Arc::new(Index::open(":memory:").unwrap());
    let writers: Vec<_> = (0..4)
        .map(|t| {
            let idx = Arc::clone(&idx);
            thread::spawn(move || {
                for i in 0..50 {
                    idx.upsert(t * 1000 + i, &format!("worker {t} document {i} shared")).unwrap();
                    idx.search_bm25("shared worker", 5).unwrap();
                }
            })
        })
        .collect();
    for w in writers {
        w.join().unwrap();
    }
    assert_eq!(idx.len().unwrap(), 200);
}
