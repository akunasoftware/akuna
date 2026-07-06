use super::*;
use crate::ml::backend::{Backend, cpu_device};
use burn::tensor::Tensor;

#[test]
fn api_options_default_uses_minilm_l12() {
    assert_eq!(
        TextEmbedderOptions::default().model,
        EmbeddingModel::MiniLmL12
    );
}

#[test]
fn api_model_metadata_returns_repo_ids() {
    assert_eq!(
        EmbeddingModel::BgeSmallEnV15.repo_id(),
        "BAAI/bge-small-en-v1.5"
    );
    assert_eq!(
        EmbeddingModel::BgeBaseEnV15.repo_id(),
        "BAAI/bge-base-en-v1.5"
    );
    assert_eq!(
        EmbeddingModel::BgeLargeEnV15.repo_id(),
        "BAAI/bge-large-en-v1.5"
    );
    assert_eq!(
        EmbeddingModel::AllMpnetBaseV2.repo_id(),
        "sentence-transformers/all-mpnet-base-v2"
    );
    assert_eq!(EmbeddingModel::BgeM3.repo_id(), "BAAI/bge-m3");
}

#[tokio::test]
async fn model_minilm_l6_can_embed_text() {
    let model = TextEmbedder::new_on(
        cpu_device(),
        TextEmbedderOptions {
            model: EmbeddingModel::MiniLmL6,
            cache_dir: None,
        },
    )
    .await
    .expect("model should load");

    let single = model
        .embed("Hello world")
        .expect("single embed should work");
    assert!(!single.is_empty());
}

#[test]
fn util_batch_size_default_caps_large_batches() {
    let batch_size = resolve_batch_size(128, None, DEFAULT_BATCH_SIZE)
        .expect("default batch size should work");
    assert_eq!(batch_size, DEFAULT_BATCH_SIZE);
}

#[test]
fn util_batch_size_default_uses_document_count_when_small() {
    let batch_size = resolve_batch_size(4, None, DEFAULT_BATCH_SIZE)
        .expect("default batch size should work");
    assert_eq!(batch_size, 4);
}

#[test]
fn util_batch_size_validate_rejects_zero() {
    let error = resolve_batch_size(1, Some(0), DEFAULT_BATCH_SIZE)
        .expect_err("zero batch size should fail");
    assert!(
        error
            .to_string()
            .contains("batch size must be greater than zero")
    );
}

#[test]
fn util_tensor_rows_extract_returns_rows() {
    let device = cpu_device();
    let embeddings =
        Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);

    let rows = tensor2_to_rows_f32(
        embeddings,
        "failed to read embedding output tensor",
    )
    .expect("rows should extract");
    assert_eq!(rows, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
}
