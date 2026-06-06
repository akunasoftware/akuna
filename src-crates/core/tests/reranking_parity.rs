//! Parity test comparing akuna-core reranker scores against FlagEmbedding run
//! via `scripts/reference_rerank.py`.
//!
//! Downloads a checkpoint and spawns a Python reference, so the test is marked
//! `#[ignore]`. Run with:
//!
//! ```sh
//! cargo test -p akuna-core --features full --test reranking_parity -- \
//!     --nocapture --ignored
//! ```
#![cfg(feature = "reranking")]

#[path = "common.rs"]
mod common;

use anyhow::Result;

use akuna_core::reranking::{TextReranker, TextRerankerOptions};

#[tokio::test]
#[ignore = "downloads model and runs Python transformers reference"]
async fn parity_bge_base_scores_match_flag_embedding() {
    let pairs = vec![
        (
            "Rust machine learning".to_string(),
            "Burn is a deep learning framework for Rust".to_string(),
        ),
        (
            "Rust machine learning".to_string(),
            "Bananas are yellow".to_string(),
        ),
    ];
    let model = TextReranker::new(TextRerankerOptions::default())
        .await
        .expect("model should load");
    let actual = model
        .score_batch(&pairs, Some(2))
        .expect("Burn reranker should score pairs");
    let expected = reference_scores("BAAI/bge-reranker-base", &pairs)
        .expect("reference scores should compute");

    // Floor measured vs FlagEmbedding: worst score delta 5.7e-6 across pairs.
    assert_scores_close(&actual, &expected, 5e-5);
}

fn reference_scores(
    model: &str,
    pairs: &[(String, String)],
) -> Result<Vec<f32>> {
    common::run_uv_script_json(
        "reference_rerank.py",
        &["--model", model],
        &pairs,
    )
}

fn assert_scores_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        let delta = (actual - expected).abs();
        assert!(
            delta <= tolerance,
            "score delta {delta} exceeded tolerance {tolerance}: actual {actual}, expected {expected}"
        );
    }
}
