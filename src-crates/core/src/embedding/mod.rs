//! Dense text embeddings.
//!
//! Generate `Vec<f32>` embeddings for text or batches of text. Select a
//! checkpoint via `EmbeddingModel` (defaults to `MiniLmL12`). Inference is
//! blocking; wrap it in `tokio::task::spawn_blocking` from async contexts.
//!
//! # Example
//!
//! ```rust,no_run
//! use akuna_core::embedding::{EmbeddingModel, TextEmbedding, TextEmbeddingOptions};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let model = TextEmbedding::new(TextEmbeddingOptions {
//!         model: EmbeddingModel::MiniLmL12,
//!         ..Default::default()
//!     })
//!     .await?;
//!
//!     // `embed` is blocking. In production, wrap heavy inference in
//!     // `tokio::task::spawn_blocking` to avoid stalling async workers.
//!     let single = model.embed("Hello world")?;
//!     assert!(!single.is_empty());
//!
//!     let batch = model.embed_batch(&["Hello world", "Rust embeddings"], None)?;
//!     assert_eq!(batch.len(), 2);
//!
//!     Ok(())
//! }
//! ```

mod models;

use std::path::PathBuf;

use anyhow::{Context, Result};
use burn_dispatch::DispatchDevice;

use crate::embedding::models::bert::{
    BertEmbeddingModel, PoolingStrategy, load_pretrained_bert_embedding,
};
use crate::embedding::models::mpnet::{
    MpnetEmbeddingModel, load_pretrained_mpnet_embedding,
};
use crate::embedding::models::xlm_roberta::{
    XlmRobertaEmbeddingModel, load_pretrained_xlm_roberta_embedding,
};
use crate::ml::backend::{self, Backend};
use crate::ml::{resolve_batch_size, tensor2_to_rows_f32};

/// Default batch size when the caller does not supply one.
const DEFAULT_BATCH_SIZE: usize = 32;

/// Supported embedding model checkpoints.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EmbeddingModel {
    /// `sentence-transformers/all-MiniLM-L6-v2`.
    MiniLmL6,
    /// `sentence-transformers/all-MiniLM-L12-v2`.
    #[default]
    MiniLmL12,
    /// `BAAI/bge-small-en-v1.5`.
    BgeSmallEnV15,
    /// `BAAI/bge-base-en-v1.5`.
    BgeBaseEnV15,
    /// `BAAI/bge-large-en-v1.5`.
    BgeLargeEnV15,
    /// `sentence-transformers/all-mpnet-base-v2`.
    AllMpnetBaseV2,
    /// `BAAI/bge-m3` dense embeddings.
    BgeM3,
}

impl EmbeddingModel {
    /// Hugging Face repository for this checkpoint.
    fn repo_id(self) -> &'static str {
        match self {
            Self::MiniLmL6 => "sentence-transformers/all-MiniLM-L6-v2",
            Self::MiniLmL12 => "sentence-transformers/all-MiniLM-L12-v2",
            Self::BgeSmallEnV15 => "BAAI/bge-small-en-v1.5",
            Self::BgeBaseEnV15 => "BAAI/bge-base-en-v1.5",
            Self::BgeLargeEnV15 => "BAAI/bge-large-en-v1.5",
            Self::AllMpnetBaseV2 => "sentence-transformers/all-mpnet-base-v2",
            Self::BgeM3 => "BAAI/bge-m3",
        }
    }
}

#[derive(Debug)]
enum LoadedEmbeddingModel {
    Bert(BertEmbeddingModel<Backend>),
    Mpnet(MpnetEmbeddingModel<Backend>),
    XlmRoberta(XlmRobertaEmbeddingModel<Backend>),
}

/// Options for [`TextEmbedding`].
#[derive(Debug, Clone, Default)]
pub struct TextEmbeddingOptions {
    /// Which embedding checkpoint to load.
    pub model: EmbeddingModel,
    /// Optional Hugging Face cache directory override.
    pub cache_dir: Option<PathBuf>,
}

/// Text embedding model.
pub struct TextEmbedding {
    model: LoadedEmbeddingModel,
    model_kind: EmbeddingModel,
    device: DispatchDevice,
}

impl TextEmbedding {
    /// Loads the embedding model from `options` onto the default device.
    pub async fn new(options: TextEmbeddingOptions) -> Result<Self> {
        Self::new_on(backend::active_device(), options).await
    }

    /// Loads the embedding model from `options` onto a specific device.
    pub(crate) async fn new_on(
        device: DispatchDevice,
        options: TextEmbeddingOptions,
    ) -> Result<Self> {
        let model_kind = options.model;
        let repo_id = model_kind.repo_id();
        // BERT-family models differ only by checkpoint + pooling; MiniLM uses
        // mean pooling, BGE uses CLS. MPNet and XLM-RoBERTa are their own
        // backbones with a single checkpoint each.
        let model: LoadedEmbeddingModel = match model_kind {
            EmbeddingModel::MiniLmL6 | EmbeddingModel::MiniLmL12 => {
                LoadedEmbeddingModel::Bert(
                    load_pretrained_bert_embedding(
                        &device,
                        repo_id,
                        PoolingStrategy::Mean,
                        options.cache_dir,
                    )
                    .await?,
                )
            }
            EmbeddingModel::BgeSmallEnV15
            | EmbeddingModel::BgeBaseEnV15
            | EmbeddingModel::BgeLargeEnV15 => LoadedEmbeddingModel::Bert(
                load_pretrained_bert_embedding(
                    &device,
                    repo_id,
                    PoolingStrategy::Cls,
                    options.cache_dir,
                )
                .await?,
            ),
            EmbeddingModel::AllMpnetBaseV2 => LoadedEmbeddingModel::Mpnet(
                load_pretrained_mpnet_embedding(
                    &device,
                    repo_id,
                    options.cache_dir,
                )
                .await?,
            ),
            EmbeddingModel::BgeM3 => LoadedEmbeddingModel::XlmRoberta(
                load_pretrained_xlm_roberta_embedding(
                    &device,
                    repo_id,
                    options.cache_dir,
                )
                .await?,
            ),
        };

        Ok(Self {
            model,
            model_kind,
            device,
        })
    }

    /// Embeds a batch of inputs with an optional prompt, one vector per input.
    fn embed_batch_inner(
        &self,
        inputs: &[&str],
        batch_size: Option<usize>,
        prompt: Option<&str>,
    ) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size =
            resolve_batch_size(inputs.len(), batch_size, DEFAULT_BATCH_SIZE)?;

        let mut embeddings = Vec::with_capacity(inputs.len());
        for batch in inputs.chunks(batch_size) {
            let batch_embeddings = match &self.model {
                LoadedEmbeddingModel::Bert(model) => {
                    model.encode(batch, prompt, &self.device)?
                }
                LoadedEmbeddingModel::Mpnet(model) => {
                    model.encode(batch, prompt, &self.device)?
                }
                LoadedEmbeddingModel::XlmRoberta(model) => {
                    model.encode(batch, prompt, &self.device)?
                }
            };
            embeddings.extend(tensor2_to_rows_f32(
                batch_embeddings,
                "failed to read embedding output tensor",
            )?);
        }

        Ok(embeddings)
    }

    /// Embeds a single document and returns one embedding vector.
    pub fn embed(&self, document: impl AsRef<str>) -> Result<Vec<f32>> {
        self.embed_with_prompt(document, None)
    }

    /// Embeds a single document with an optional input prompt.
    pub fn embed_with_prompt(
        &self,
        document: impl AsRef<str>,
        prompt: Option<&str>,
    ) -> Result<Vec<f32>> {
        let mut embeddings =
            self.embed_batch_inner(&[document.as_ref()], None, prompt)?;
        embeddings
            .pop()
            .context("expected one embedding for a single input document")
    }

    /// Embeds documents in batches and returns one vector per input string.
    pub fn embed_batch<S: AsRef<str>>(
        &self,
        documents: &[S],
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed_batch_with_prompt(documents, batch_size, None)
    }

    /// Embeds documents in batches with an optional input prompt.
    pub fn embed_batch_with_prompt<S: AsRef<str>>(
        &self,
        documents: &[S],
        batch_size: Option<usize>,
        prompt: Option<&str>,
    ) -> Result<Vec<Vec<f32>>> {
        let inputs = documents.iter().map(AsRef::as_ref).collect::<Vec<_>>();
        self.embed_batch_inner(&inputs, batch_size, prompt)
    }

    /// Returns the loaded embedding checkpoint.
    pub fn model(&self) -> EmbeddingModel {
        self.model_kind
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::backend::{Backend, cpu_device};
    use burn::tensor::Tensor;
    use std::sync::OnceLock;
    use tokio::sync::Mutex;

    static LIVE_MODEL_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    #[test]
    fn api_model_metadata_returns_bge_repo_ids() {
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
    fn api_options_default_uses_minilm_l12() {
        assert_eq!(
            TextEmbeddingOptions::default().model,
            EmbeddingModel::MiniLmL12
        );
    }

    #[tokio::test]
    async fn model_bge_small_embed_returns_document_and_query_vectors() {
        let _guard = live_model_test_lock().lock().await;
        let model = TextEmbedding::new(TextEmbeddingOptions {
            model: EmbeddingModel::BgeSmallEnV15,
            ..Default::default()
        })
        .await
        .expect("model should load");

        let document = model
            .embed("Hello world")
            .expect("document embed should work");
        let query =
            model.embed("Hello world").expect("query embed should work");

        assert_eq!(document.len(), 384);
        assert_eq!(query.len(), 384);
    }

    #[tokio::test]
    async fn model_minilm_l6_backend_supports_i32_indices() {
        let _guard = live_model_test_lock().lock().await;
        let model = TextEmbedding::new_on(
            cpu_device(),
            TextEmbeddingOptions {
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

    #[tokio::test]
    async fn model_minilm_l6_embed_returns_vectors() {
        let _guard = live_model_test_lock().lock().await;
        let model = TextEmbedding::new(TextEmbeddingOptions {
            model: EmbeddingModel::MiniLmL6,
            ..Default::default()
        })
        .await
        .expect("model should load");

        let single = model
            .embed("Hello world")
            .expect("single embed should work");
        assert!(!single.is_empty());

        let batch = model
            .embed_batch(&["Hello world", "Rust embeddings"], None)
            .expect("batch embed should work");
        assert_eq!(batch.len(), 2);
        assert!(batch.iter().all(|embedding| !embedding.is_empty()));
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
        let embeddings = Tensor::<Backend, 2>::from_floats(
            [[1.0, 2.0], [3.0, 4.0]],
            &device,
        );

        let rows = tensor2_to_rows_f32(
            embeddings,
            "failed to read embedding output tensor",
        )
        .expect("rows should extract");
        assert_eq!(rows, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    fn live_model_test_lock() -> &'static Mutex<()> {
        LIVE_MODEL_TEST_LOCK.get_or_init(|| Mutex::new(()))
    }
}
