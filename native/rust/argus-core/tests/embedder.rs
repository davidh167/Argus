use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use argus_core::{Embedder, Error};

/// ARGUS_MODEL, or the Ollama nomic-embed-text blob. None (test skipped) if neither exists.
fn model() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ARGUS_MODEL") {
        return Some(PathBuf::from(p));
    }
    let blob = PathBuf::from(std::env::var("HOME").ok()?)
        .join(".ollama/models/blobs/sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6");
    blob.exists().then_some(blob)
}

fn open() -> Option<Embedder> {
    let Some(path) = model() else {
        eprintln!("skipping: no embedding model (set ARGUS_MODEL)");
        return None;
    };
    Some(Embedder::open(path.to_str().unwrap(), 0).unwrap())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn vectors_have_the_models_dimension_and_unit_length() {
    let Some(e) = open() else { return };
    assert_eq!(e.dim(), 768);
    let v = e.embed(&["distributed systems design"]).unwrap();
    assert_eq!((v.len(), v.dim()), (1, 768));
    assert!((dot(v.row(0), v.row(0)) - 1.0).abs() < 1e-4);
    assert!(!v.was_truncated(0));
}

#[test]
fn related_phrases_are_closer_than_unrelated_ones() {
    let Some(e) = open() else { return };
    let v = e.embed(&["machine learning", "deep learning", "pastry baking"]).unwrap();
    assert!(dot(v.row(0), v.row(1)) > dot(v.row(0), v.row(2)));
}

#[test]
fn a_batch_matches_one_at_a_time() {
    let Some(e) = open() else { return };
    let texts = ["PyTorch", "C++", "retrieval augmented generation pipelines"];
    let batch = e.embed(&texts).unwrap();
    for (i, t) in texts.iter().enumerate() {
        let single = e.embed(&[*t]).unwrap();
        assert!(dot(single.row(0), batch.row(i)) > 0.9999, "{t}");
    }
}

#[test]
fn over_long_input_is_truncated_and_flagged() {
    let Some(e) = open() else { return };
    let long: String = (0..6000).map(|i| format!("word{} ", i % 50)).collect();
    let v = e.embed(&[long.as_str(), "short"]).unwrap();
    assert!(v.was_truncated(0));
    assert!(!v.was_truncated(1));
}

#[test]
fn a_missing_model_is_an_error_not_a_panic() {
    assert!(matches!(Embedder::open("/no/such/model.gguf", 0), Err(Error::Model(_))));
    assert_eq!(Embedder::open("bad\0path", 0).err(), Some(Error::InvalidPath));
}

#[test]
fn a_shared_embedder_is_safe_across_threads() {
    let Some(e) = open() else { return };
    let e = Arc::new(e);
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let e = Arc::clone(&e);
            thread::spawn(move || {
                let v = e.embed(&[format!("worker {t} phrase")]).unwrap();
                assert!((dot(v.row(0), v.row(0)) - 1.0).abs() < 1e-4);
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}
