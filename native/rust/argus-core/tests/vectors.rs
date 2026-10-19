use std::path::PathBuf;

use argus_core::{Embedder, Error, Index};

fn unit(v: [f32; 4]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / n).collect()
}

fn ids(hits: &[argus_core::Hit]) -> Vec<i64> {
    hits.iter().map(|h| h.id).collect()
}

#[test]
fn dense_search_ranks_by_cosine() {
    let idx = Index::open_with_vectors(":memory:", 4).unwrap();
    idx.upsert_vector(1, &unit([1.0, 0.0, 0.0, 0.0])).unwrap();
    idx.upsert_vector(2, &unit([1.0, 1.0, 0.0, 0.0])).unwrap();
    idx.upsert_vector(3, &unit([0.0, 0.0, 1.0, 0.0])).unwrap();
    let hits = idx.search_dense(&unit([1.0, 0.1, 0.0, 0.0]), 10).unwrap();
    assert_eq!(ids(&hits), vec![1, 2, 3]);
}

#[test]
fn replace_and_remove_update_the_vector_index() {
    let idx = Index::open_with_vectors(":memory:", 4).unwrap();
    idx.upsert_vector(1, &unit([1.0, 0.0, 0.0, 0.0])).unwrap();
    idx.upsert_vector(2, &unit([0.0, 1.0, 0.0, 0.0])).unwrap();
    idx.upsert_vector(1, &unit([0.0, 0.0, 1.0, 0.0])).unwrap();
    assert_eq!(idx.search_dense(&unit([0.0, 0.0, 1.0, 0.0]), 1).unwrap()[0].id, 1);
    idx.remove(1).unwrap();
    assert!(!ids(&idx.search_dense(&unit([0.0, 0.0, 1.0, 0.0]), 10).unwrap()).contains(&1));
}

#[test]
fn hybrid_fuses_keyword_and_dense_rankings() {
    let idx = Index::open_with_vectors(":memory:", 4).unwrap();
    idx.upsert(1, "retrieval retrieval retrieval").unwrap();
    idx.upsert(2, "cooking recipes and baking").unwrap();
    idx.upsert(3, "retrieval systems for large scale enterprise search and many other unrelated words here").unwrap();
    idx.upsert_vector(1, &unit([0.0, 0.0, 0.0, 1.0])).unwrap();
    idx.upsert_vector(2, &unit([0.9, 0.3, 0.0, 0.0])).unwrap();
    idx.upsert_vector(3, &unit([1.0, 0.0, 0.0, 0.0])).unwrap();
    let hits = idx.search_hybrid("retrieval", &unit([1.0, 0.0, 0.0, 0.0]), 10).unwrap();
    assert_eq!(ids(&hits), vec![3, 1, 2]);
}

#[test]
fn wrong_dimension_and_missing_vector_index_are_errors() {
    let idx = Index::open_with_vectors(":memory:", 4).unwrap();
    assert_eq!(idx.upsert_vector(1, &[1.0, 2.0]).err(), Some(Error::InvalidArgument));
    assert_eq!(idx.search_dense(&[1.0], 5).err(), Some(Error::InvalidArgument));
    let plain = Index::open(":memory:").unwrap();
    assert_eq!(plain.upsert_vector(1, &[1.0, 0.0, 0.0, 0.0]).err(), Some(Error::InvalidArgument));
    assert_eq!(Index::open_with_vectors(":memory:", 0).err(), Some(Error::InvalidArgument));
}

#[test]
fn reopening_rebuilds_the_index_and_rejects_another_dimension() {
    let path = std::env::temp_dir().join(format!("argus_rust_vec_{}.db", std::process::id()));
    let p = path.to_str().unwrap();
    {
        let idx = Index::open_with_vectors(p, 4).unwrap();
        idx.upsert_vector(5, &unit([1.0, 0.0, 0.0, 0.0])).unwrap();
        idx.upsert_vector(6, &unit([0.0, 1.0, 0.0, 0.0])).unwrap();
    }
    {
        let idx = Index::open_with_vectors(p, 4).unwrap();
        assert_eq!(idx.search_dense(&unit([0.0, 1.0, 0.0, 0.0]), 1).unwrap()[0].id, 6);
    }
    assert_eq!(Index::open_with_vectors(p, 5).err(), Some(Error::InvalidArgument));
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{p}{suffix}"));
    }
}

/// End to end: embed real text with the model, store it, and retrieve it by meaning and by hybrid.
#[test]
fn embed_then_search_end_to_end() {
    let model = std::env::var("ARGUS_MODEL").map(PathBuf::from).ok().or_else(|| {
        let blob = PathBuf::from(std::env::var("HOME").ok()?)
            .join(".ollama/models/blobs/sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6");
        blob.exists().then_some(blob)
    });
    let Some(model) = model else {
        eprintln!("skipping: no embedding model (set ARGUS_MODEL)");
        return;
    };
    let embedder = Embedder::open(model.to_str().unwrap(), 0).unwrap();
    let idx = Index::open_with_vectors(":memory:", embedder.dim()).unwrap();
    let docs = [
        "search_document: Machine learning engineer building retrieval and RAG pipelines for LLM agents",
        "search_document: Pastry chef specializing in laminated dough and French baking",
        "search_document: Embedded firmware engineer writing C for microcontrollers",
    ];
    let vecs = embedder.embed(&docs).unwrap();
    for (i, d) in docs.iter().enumerate() {
        idx.upsert(i as i64, d).unwrap();
        idx.upsert_vector(i as i64, vecs.row(i)).unwrap();
    }
    let q = embedder.embed(&["search_query: LLM retrieval engineer"]).unwrap();
    assert_eq!(idx.search_dense(q.row(0), 3).unwrap()[0].id, 0);
    assert_eq!(idx.search_hybrid("retrieval", q.row(0), 3).unwrap()[0].id, 0);
}
