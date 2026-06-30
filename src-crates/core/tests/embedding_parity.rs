//! Parity tests comparing akuna-core embedding output against
//! `sentence-transformers` run via `scripts/reference_embeddings.py`.
//!
//! Each test downloads a checkpoint and spawns a Python reference, so all are
//! marked `#[ignore]`. Run one with:
//!
//! ```sh
//! cargo test -p akuna-core --features full --test embedding_parity -- \
//!     --nocapture --ignored parity_bge_small_document_matches_sentence_transformers
//! ```
#![cfg(feature = "embedding")]

#[path = "common.rs"]
mod common;

use std::sync::OnceLock;

use anyhow::Result;
use tokio::sync::Mutex;

use akuna_core::embedding::{
    EmbeddingModel, TextEmbedding, TextEmbeddingOptions,
};

/// Per-process lock serialising live-model tests that share the download
/// cache and WGPU device.
static LIVE_MODEL_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

const BGE_QUERY_PROMPT: &str =
    "Represent this sentence for searching relevant passages: ";

#[ignore = "downloads model and runs Python reference"]
#[tokio::test]
async fn parity_bge_base_document_matches_sentence_transformers() {
    assert_model_matches_sentence_transformers(
        EmbeddingModel::BgeBaseEnV15,
        "BAAI/bge-base-en-v1.5",
        ReferenceInputKind::Document,
    )
    .await;
}

#[ignore = "downloads model and runs Python reference"]
#[tokio::test]
async fn parity_bge_base_query_matches_sentence_transformers() {
    assert_model_matches_sentence_transformers(
        EmbeddingModel::BgeBaseEnV15,
        "BAAI/bge-base-en-v1.5",
        ReferenceInputKind::Query,
    )
    .await;
}

#[ignore = "downloads model and runs Python reference"]
#[tokio::test]
async fn parity_bge_large_document_matches_sentence_transformers() {
    assert_model_matches_sentence_transformers(
        EmbeddingModel::BgeLargeEnV15,
        "BAAI/bge-large-en-v1.5",
        ReferenceInputKind::Document,
    )
    .await;
}

#[ignore = "downloads model and runs Python reference"]
#[tokio::test]
async fn parity_bge_large_query_matches_sentence_transformers() {
    assert_model_matches_sentence_transformers(
        EmbeddingModel::BgeLargeEnV15,
        "BAAI/bge-large-en-v1.5",
        ReferenceInputKind::Query,
    )
    .await;
}

#[ignore = "downloads model and runs Python reference"]
#[tokio::test]
async fn parity_bge_small_document_matches_sentence_transformers() {
    assert_model_matches_sentence_transformers(
        EmbeddingModel::BgeSmallEnV15,
        "BAAI/bge-small-en-v1.5",
        ReferenceInputKind::Document,
    )
    .await;
}

#[ignore = "downloads model and runs Python reference"]
#[tokio::test]
async fn parity_bge_small_query_matches_sentence_transformers() {
    assert_model_matches_sentence_transformers(
        EmbeddingModel::BgeSmallEnV15,
        "BAAI/bge-small-en-v1.5",
        ReferenceInputKind::Query,
    )
    .await;
}

#[ignore = "downloads model and runs Python reference"]
#[tokio::test]
async fn parity_bge_small_query_with_prompt_matches_sentence_transformers() {
    assert_model_matches_sentence_transformers_with_prompt(
        EmbeddingModel::BgeSmallEnV15,
        "BAAI/bge-small-en-v1.5",
        ReferenceInputKind::Query,
        BGE_QUERY_PROMPT,
    )
    .await;
}

#[ignore = "downloads model and runs Python reference"]
#[tokio::test]
async fn parity_minilm_l12_document_matches_sentence_transformers() {
    assert_model_matches_sentence_transformers(
        EmbeddingModel::MiniLmL12,
        "sentence-transformers/all-MiniLM-L12-v2",
        ReferenceInputKind::Document,
    )
    .await;
}

#[ignore = "downloads model and runs Python reference"]
#[tokio::test]
async fn parity_minilm_l12_query_matches_sentence_transformers() {
    assert_model_matches_sentence_transformers(
        EmbeddingModel::MiniLmL12,
        "sentence-transformers/all-MiniLM-L12-v2",
        ReferenceInputKind::Query,
    )
    .await;
}

#[ignore = "downloads model and runs Python reference"]
#[tokio::test]
async fn parity_minilm_l6_document_matches_sentence_transformers() {
    assert_model_matches_sentence_transformers(
        EmbeddingModel::MiniLmL6,
        "sentence-transformers/all-MiniLM-L6-v2",
        ReferenceInputKind::Document,
    )
    .await;
}

#[ignore = "downloads model and runs Python reference"]
#[tokio::test]
async fn parity_minilm_l6_query_matches_sentence_transformers() {
    assert_model_matches_sentence_transformers(
        EmbeddingModel::MiniLmL6,
        "sentence-transformers/all-MiniLM-L6-v2",
        ReferenceInputKind::Query,
    )
    .await;
}

#[ignore = "downloads model and runs Python reference"]
#[tokio::test]
async fn parity_mpnet_base_document_matches_sentence_transformers() {
    assert_model_matches_sentence_transformers(
        EmbeddingModel::AllMpnetBaseV2,
        "sentence-transformers/all-mpnet-base-v2",
        ReferenceInputKind::Document,
    )
    .await;
}

#[ignore = "downloads model and runs Python reference"]
#[tokio::test]
async fn parity_mpnet_base_query_matches_sentence_transformers() {
    assert_model_matches_sentence_transformers(
        EmbeddingModel::AllMpnetBaseV2,
        "sentence-transformers/all-mpnet-base-v2",
        ReferenceInputKind::Query,
    )
    .await;
}

#[tokio::test]
#[ignore = "BGE-M3 is too large for the default WGPU parity suite"]
async fn parity_bge_m3_document_matches_sentence_transformers() {
    assert_model_matches_sentence_transformers(
        EmbeddingModel::BgeM3,
        "BAAI/bge-m3",
        ReferenceInputKind::Document,
    )
    .await;
}

#[tokio::test]
#[ignore = "BGE-M3 is too large for the default WGPU parity suite"]
async fn parity_bge_m3_query_matches_sentence_transformers() {
    assert_model_matches_sentence_transformers(
        EmbeddingModel::BgeM3,
        "BAAI/bge-m3",
        ReferenceInputKind::Query,
    )
    .await;
}

#[derive(Debug, Clone, Copy)]
enum ReferenceInputKind {
    Document,
    Query,
}

impl ReferenceInputKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Document => "document",
            Self::Query => "query",
        }
    }
}

async fn assert_model_matches_sentence_transformers(
    model: EmbeddingModel,
    reference_model: &str,
    input_kind: ReferenceInputKind,
) {
    let _guard = live_model_test_lock().lock().await;
    assert_model_matches_sentence_transformers_for_texts(
        model,
        reference_model,
        input_kind,
        parity_texts(),
    )
    .await;
    assert_model_matches_sentence_transformers_for_texts(
        model,
        reference_model,
        input_kind,
        long_parity_texts(),
    )
    .await;
}

async fn assert_model_matches_sentence_transformers_with_prompt(
    model: EmbeddingModel,
    reference_model: &str,
    input_kind: ReferenceInputKind,
    prompt: &str,
) {
    let _guard = live_model_test_lock().lock().await;
    assert_model_matches_sentence_transformers_for_texts_with_prompt(
        model,
        reference_model,
        input_kind,
        parity_texts(),
        Some(prompt),
    )
    .await;
}

async fn assert_model_matches_sentence_transformers_for_texts(
    model: EmbeddingModel,
    reference_model: &str,
    input_kind: ReferenceInputKind,
    texts: Vec<String>,
) {
    assert_model_matches_sentence_transformers_for_texts_with_prompt(
        model,
        reference_model,
        input_kind,
        texts,
        None,
    )
    .await;
}

async fn assert_model_matches_sentence_transformers_for_texts_with_prompt(
    model: EmbeddingModel,
    reference_model: &str,
    input_kind: ReferenceInputKind,
    texts: Vec<String>,
    prompt: Option<&str>,
) {
    let model = TextEmbedding::new(TextEmbeddingOptions {
        model,
        ..Default::default()
    })
    .await
    .expect("model should load");
    let actual = match input_kind {
        ReferenceInputKind::Document => model
            .embed_batch_with_prompt(&texts, Some(2), prompt)
            .expect("Burn document embeddings should work"),
        ReferenceInputKind::Query => model
            .embed_batch_with_prompt(&texts, Some(2), prompt)
            .expect("Burn query embeddings should work"),
    };
    let expected = reference_embeddings(
        reference_model,
        input_kind.as_str(),
        &texts,
        prompt,
    )
    .expect("reference embeddings should work");

    assert_embedding_batches_close(
        &actual,
        &expected,
        &texts,
        model.model(),
        input_kind,
        max_delta_tolerance(model.model()),
        0.9999,
    );
}

fn max_delta_tolerance(model: EmbeddingModel) -> f32 {
    // Floors measured against sentence-transformers (worst component delta over
    // all parity texts/input kinds): the BERT-family models all land ≤ 6.1e-7,
    // while BgeM3 (24-layer XLM-R-large) accumulates ~30× more at 1.8e-5. These
    // bounds sit a small factor above the observed worst — tight enough that any
    // real regression trips them, with headroom only for f32 op-ordering drift.
    match model {
        EmbeddingModel::BgeM3 => 1e-4,
        EmbeddingModel::MiniLmL6
        | EmbeddingModel::MiniLmL12
        | EmbeddingModel::BgeSmallEnV15
        | EmbeddingModel::BgeBaseEnV15
        | EmbeddingModel::BgeLargeEnV15
        | EmbeddingModel::AllMpnetBaseV2 => 5e-6,
    }
}

fn parity_texts() -> Vec<String> {
    vec![
        "Hello world".to_string(),
        "Rust embeddings".to_string(),
        "Semantic search: fast, accurate, and simple.".to_string(),
        "  padded input with leading and trailing spaces  ".to_string(),
        "Numbers 12345, symbols !?., and mixed CASE.".to_string(),
        "emoji rocket and unicode cafe".to_string(),
    ]
    .into_iter()
    .chain([
        "Machine learning in Rust.".to_string(),
        "Apprendimento automatico in Rust.".to_string(),
        "Rust での機械学習".to_string(),
    ])
    .collect()
}

fn long_parity_texts() -> Vec<String> {
    let sentence = "Burn embeddings should match sentence-transformers even when tokenizer truncation is required. ";
    vec![sentence.repeat(128)]
}

fn live_model_test_lock() -> &'static Mutex<()> {
    LIVE_MODEL_TEST_LOCK.get_or_init(|| Mutex::new(()))
}

fn reference_embeddings(
    model: &str,
    kind: &str,
    texts: &[String],
    prompt: Option<&str>,
) -> Result<Vec<Vec<f32>>> {
    let mut args = vec!["--model", model, "--kind", kind];
    if let Some(prompt) = prompt {
        args.extend(["--prompt", prompt]);
    }
    common::run_uv_script_json("reference_embeddings.py", &args, &texts)
}

fn assert_embedding_batches_close(
    actual: &[Vec<f32>],
    expected: &[Vec<f32>],
    texts: &[String],
    model: EmbeddingModel,
    input_kind: ReferenceInputKind,
    tolerance: f32,
    min_cosine_similarity: f32,
) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.len(),
            expected.len(),
            "embedding width mismatch for {model:?} {input_kind:?} input {index}: {:?}",
            texts.get(index)
        );
        let max_delta = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_delta <= tolerance,
            "max embedding delta {max_delta} exceeded tolerance {tolerance} for {model:?} {input_kind:?} input {index}: {:?}",
            texts.get(index)
        );
        let cosine_similarity = cosine_similarity(actual, expected);
        assert!(
            cosine_similarity >= min_cosine_similarity,
            "cosine similarity {cosine_similarity} fell below {min_cosine_similarity} for {model:?} {input_kind:?} input {index}: {:?}",
            texts.get(index)
        );
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let dot_product = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm =
        right.iter().map(|value| value * value).sum::<f32>().sqrt();

    dot_product / (left_norm * right_norm)
}
