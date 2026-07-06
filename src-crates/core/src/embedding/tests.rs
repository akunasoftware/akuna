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

#[test]
fn api_model_serialization_stable() -> std::result::Result<(), serde_json::Error>
{
    let cases = [
        (EmbeddingModel::MiniLmL6, "mini_lm_l6"),
        (EmbeddingModel::MiniLmL12, "mini_lm_l12"),
        (EmbeddingModel::BgeSmallEnV15, "bge_small_en_v15"),
        (EmbeddingModel::BgeBaseEnV15, "bge_base_en_v15"),
        (EmbeddingModel::BgeLargeEnV15, "bge_large_en_v15"),
        (EmbeddingModel::AllMpnetBaseV2, "all_mpnet_base_v2"),
        (EmbeddingModel::BgeM3, "bge_m3"),
    ];

    for (model, key) in cases {
        let encoded = format!("\"{key}\"");
        assert_eq!(serde_json::to_string(&model)?, encoded);
        assert_eq!(serde_json::from_str::<EmbeddingModel>(&encoded)?, model);
    }

    Ok(())
}

#[test]
fn model_minilm_l6_can_embed_text() {
    crate::testkit::run_with_model_stack(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?
            .block_on(async {
                let model = TextEmbedder::new_on(
                    cpu_device(),
                    TextEmbedderOptions {
                        model: EmbeddingModel::MiniLmL6,
                        cache_dir: None,
                    },
                )
                .await?;

                let single = model.embed("Hello world")?;
                assert!(!single.is_empty());
                Ok(())
            })
    })
    .expect("model stack should run");
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
