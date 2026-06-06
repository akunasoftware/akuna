//! Dense text embeddings built with Burn.
//!
//! Generate `Vec<f32>` embeddings for text or batches of text. Inference is
//! blocking; wrap it in `tokio::task::spawn_blocking` when calling from async
//! contexts.
//!
//! # Models
//!
//! Select a checkpoint via [`EmbeddingModel`][crate::embedding::EmbeddingModel]
//! (defaults to `MiniLmL12`):
//!
//! - `MiniLmL6` / `MiniLmL12` — `sentence-transformers/all-MiniLM-L6-v2` / `all-MiniLM-L12-v2`
//! - `BgeSmallEnV15` / `BgeBaseEnV15` / `BgeLargeEnV15` — `BAAI/bge-small-en-v1.5` / `bge-base-en-v1.5` / `bge-large-en-v1.5`
//! - `AllMpnetBaseV2` — `sentence-transformers/all-mpnet-base-v2`
//! - `BgeM3` — `BAAI/bge-m3` (dense output only)
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
use burn::tensor::backend::Backend;
use burn_wgpu::{Wgpu, WgpuDevice};

use crate::embedding::models::bert::{
    BertEmbeddingModel, PoolingStrategy, load_pretrained_bert_embedding,
};
use crate::embedding::models::mpnet::{
    MpnetEmbeddingModel, load_pretrained_mpnet_embedding,
};
use crate::embedding::models::xlm_roberta::{
    XlmRobertaEmbeddingModel, load_pretrained_xlm_roberta_embedding,
};
use crate::ml::{resolve_batch_size, tensor2_to_rows_f32};

/// Default Burn backend.
type DefaultBackend = Wgpu;

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
    /// BGE-M3 dense embeddings only.
    ///
    /// Sparse and multi-vector outputs are separate retrieval concerns and are
    /// not exposed through this `Vec<f32>` dense embedding API.
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
enum LoadedEmbeddingModel<B: Backend> {
    Bert(BertEmbeddingModel<B>),
    Mpnet(MpnetEmbeddingModel<B>),
    XlmRoberta(XlmRobertaEmbeddingModel<B>),
}

/// Options for [`TextEmbedding`].
#[derive(Debug, Clone, Default)]
pub struct TextEmbeddingOptions {
    /// Which embedding checkpoint to load.
    pub model: EmbeddingModel,
    /// Optional Hugging Face cache directory override.
    pub cache_dir: Option<PathBuf>,
}

/// Minimal text embedding interface inspired by `fastembed-rs`.
#[derive(Debug)]
pub struct TextEmbedding<B: Backend = DefaultBackend> {
    model: LoadedEmbeddingModel<B>,
    model_kind: EmbeddingModel,
    device: B::Device,
}

impl TextEmbedding<DefaultBackend> {
    /// Loads the embedding model from `options` onto the default WGPU device.
    ///
    /// # Errors
    ///
    /// Returns an error if model weights, config, or tokenizer fail to download or load.
    pub async fn new(options: TextEmbeddingOptions) -> Result<Self> {
        let device = WgpuDevice::default();
        Self::new_with_device(&device, options).await
    }
}

impl<B> TextEmbedding<B>
where
    B: Backend,
{
    /// Loads the embedding model from `options` onto `device`.
    ///
    /// # Errors
    ///
    /// Returns an error if model weights, config, or tokenizer fail to download or load.
    pub async fn new_with_device(
        device: &B::Device,
        options: TextEmbeddingOptions,
    ) -> Result<Self> {
        let model_kind = options.model;
        let repo_id = model_kind.repo_id();
        // BERT-family models differ only by checkpoint + pooling; MiniLM uses
        // mean pooling, BGE uses CLS. MPNet and XLM-RoBERTa are their own
        // backbones with a single checkpoint each.
        let model = match model_kind {
            EmbeddingModel::MiniLmL6 | EmbeddingModel::MiniLmL12 => {
                LoadedEmbeddingModel::Bert(
                    load_pretrained_bert_embedding(
                        device,
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
                    device,
                    repo_id,
                    PoolingStrategy::Cls,
                    options.cache_dir,
                )
                .await?,
            ),
            EmbeddingModel::AllMpnetBaseV2 => LoadedEmbeddingModel::Mpnet(
                load_pretrained_mpnet_embedding(
                    device,
                    repo_id,
                    options.cache_dir,
                )
                .await?,
            ),
            EmbeddingModel::BgeM3 => LoadedEmbeddingModel::XlmRoberta(
                load_pretrained_xlm_roberta_embedding(
                    device,
                    repo_id,
                    options.cache_dir,
                )
                .await?,
            ),
        };

        Ok(Self {
            model,
            model_kind,
            device: device.clone(),
        })
    }

    /// Embeds a single document and returns one embedding vector.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`TextEmbedding::embed_with_prompt`].
    pub fn embed(&self, document: impl AsRef<str>) -> Result<Vec<f32>> {
        self.embed_with_prompt(document, None)
    }

    /// Embeds a single document with an optional input prompt.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`TextEmbedding::embed_batch_with_prompt`].
    pub fn embed_with_prompt(
        &self,
        document: impl AsRef<str>,
        prompt: Option<&str>,
    ) -> Result<Vec<f32>> {
        let document = document.as_ref();
        let documents = [document];
        let mut embeddings =
            self.embed_batch_with_prompt(documents.as_slice(), None, prompt)?;
        embeddings
            .pop()
            .context("expected one embedding for a single input document")
    }

    /// Embeds a search query with no prompt.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`TextEmbedding::embed_query_with_prompt`].
    pub fn embed_query(&self, query: impl AsRef<str>) -> Result<Vec<f32>> {
        self.embed_query_with_prompt(query, None)
    }

    /// Embeds a search query with an optional input prompt.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`TextEmbedding::embed_query_batch_with_prompt`].
    pub fn embed_query_with_prompt(
        &self,
        query: impl AsRef<str>,
        prompt: Option<&str>,
    ) -> Result<Vec<f32>> {
        let query = query.as_ref();
        let queries = [query];
        let mut embeddings = self.embed_query_batch_with_prompt(
            queries.as_slice(),
            None,
            prompt,
        )?;
        embeddings
            .pop()
            .context("expected one embedding for a single input query")
    }

    /// Embeds documents in batches and returns one vector per input string.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`TextEmbedding::embed_batch_with_prompt`].
    pub fn embed_batch<S: AsRef<str>>(
        &self,
        documents: &[S],
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed_batch_with_prompt(documents, batch_size, None)
    }

    /// Embeds documents with an optional input prompt.
    ///
    /// # Errors
    ///
    /// Returns an error if `batch_size` is `Some(0)`, or if tokenization or inference fails.
    pub fn embed_batch_with_prompt<S: AsRef<str>>(
        &self,
        documents: &[S],
        batch_size: Option<usize>,
        prompt: Option<&str>,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed_batch_inner(documents, batch_size, prompt)
    }

    /// Embeds search queries in batches with no prompt.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`TextEmbedding::embed_query_batch_with_prompt`].
    pub fn embed_query_batch<S: AsRef<str>>(
        &self,
        queries: &[S],
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed_query_batch_with_prompt(queries, batch_size, None)
    }

    /// Embeds search queries in batches with an optional input prompt.
    ///
    /// # Errors
    ///
    /// Returns an error if `batch_size` is `Some(0)`, or if tokenization or inference fails.
    pub fn embed_query_batch_with_prompt<S: AsRef<str>>(
        &self,
        queries: &[S],
        batch_size: Option<usize>,
        prompt: Option<&str>,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed_batch_inner(queries, batch_size, prompt)
    }

    fn embed_batch_inner<S: AsRef<str>>(
        &self,
        inputs: &[S],
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
            let batch_inputs =
                batch.iter().map(AsRef::as_ref).collect::<Vec<_>>();
            let batch_embeddings = match &self.model {
                LoadedEmbeddingModel::Bert(model) => {
                    model.encode(&batch_inputs, prompt, &self.device)?
                }
                LoadedEmbeddingModel::Mpnet(model) => {
                    model.encode(&batch_inputs, prompt, &self.device)?
                }
                LoadedEmbeddingModel::XlmRoberta(model) => {
                    model.encode(&batch_inputs, prompt, &self.device)?
                }
            };
            embeddings.extend(tensor2_to_rows_f32(
                batch_embeddings,
                "failed to read embedding output tensor",
            )?);
        }

        Ok(embeddings)
    }

    /// Returns the loaded embedding checkpoint.
    pub fn model(&self) -> EmbeddingModel {
        self.model_kind
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    use burn_wgpu::{Wgpu, WgpuDevice};
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
        let query = model
            .embed_query("Hello world")
            .expect("query embed should work");

        assert_eq!(document.len(), 384);
        assert_eq!(query.len(), 384);
    }

    #[tokio::test]
    async fn model_minilm_l6_backend_supports_i32_indices() {
        let _guard = live_model_test_lock().lock().await;
        let device = WgpuDevice::default();
        let model = TextEmbedding::<Wgpu<f32, i32>>::new_with_device(
            &device,
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
        let device = WgpuDevice::default();
        let embeddings = Tensor::<Wgpu<f32, i64>, 2>::from_floats(
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
