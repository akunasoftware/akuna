//! Dense text embeddings.
//!
//! Generate `Vec<f32>` embeddings for text or batches of text.
//!
//! # Example
//!
//! ```rust,no_run
//! use akuna_core::embedding::{EmbeddingModel, TextEmbedder, TextEmbedderOptions};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let model = TextEmbedder::new(TextEmbedderOptions {
//!         model: EmbeddingModel::MiniLmL12,
//!         ..Default::default()
//!     })
//!     .await?;
//!
//!     let single = model.embed("Hello world")?;
//!     assert!(!single.is_empty());
//!
//!     let batch = model.embed_batch(&["Hello world", "Rust embeddings"], None)?;
//!     assert_eq!(batch.len(), 2);
//!
//!     Ok(())
//! }
//! ```

mod error;
mod models;

#[cfg(test)]
mod tests;

use std::path::PathBuf;

use anyhow::Context;
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

pub use error::EmbeddingError;

type Result<T> = std::result::Result<T, EmbeddingError>;

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

/// Options for [`TextEmbedder`].
#[derive(Debug, Clone, Default)]
pub struct TextEmbedderOptions {
    /// Which embedding checkpoint to load.
    pub model: EmbeddingModel,
    /// Optional Hugging Face cache directory override.
    pub cache_dir: Option<PathBuf>,
}

/// Text embedding model.
pub struct TextEmbedder {
    model: LoadedEmbeddingModel,
    model_kind: EmbeddingModel,
    device: DispatchDevice,
}

impl TextEmbedder {
    /// Loads the embedding model from `options` onto the default device.
    pub async fn new(options: TextEmbedderOptions) -> Result<Self> {
        Self::new_on(backend::active_device(), options).await
    }

    /// Loads the embedding model from `options` onto a specific device.
    pub(crate) async fn new_on(
        device: DispatchDevice,
        options: TextEmbedderOptions,
    ) -> Result<Self> {
        let model_kind = options.model;
        let repo_id = model_kind.repo_id();
        // BERT-family models differ only by checkpoint + pooling; MiniLM uses
        // mean pooling, BGE uses CLS. MPNet and XLM-RoBERTa are their own
        // backbones with a single checkpoint each.
        let model: LoadedEmbeddingModel = match model_kind {
            EmbeddingModel::MiniLmL6 | EmbeddingModel::MiniLmL12 => {
                let max_length = match model_kind {
                    EmbeddingModel::MiniLmL6 => 256,
                    EmbeddingModel::MiniLmL12 => 128,
                    _ => unreachable!(),
                };
                LoadedEmbeddingModel::Bert(
                    load_pretrained_bert_embedding(
                        &device,
                        repo_id,
                        PoolingStrategy::Mean,
                        Some(max_length),
                        options.cache_dir,
                    )
                    .await
                    .map_err(EmbeddingError::load)?,
                )
            }
            EmbeddingModel::BgeSmallEnV15
            | EmbeddingModel::BgeBaseEnV15
            | EmbeddingModel::BgeLargeEnV15 => LoadedEmbeddingModel::Bert(
                load_pretrained_bert_embedding(
                    &device,
                    repo_id,
                    PoolingStrategy::Cls,
                    None,
                    options.cache_dir,
                )
                .await
                .map_err(EmbeddingError::load)?,
            ),
            EmbeddingModel::AllMpnetBaseV2 => LoadedEmbeddingModel::Mpnet(
                load_pretrained_mpnet_embedding(
                    &device,
                    repo_id,
                    384,
                    options.cache_dir,
                )
                .await
                .map_err(EmbeddingError::load)?,
            ),
            EmbeddingModel::BgeM3 => LoadedEmbeddingModel::XlmRoberta(
                load_pretrained_xlm_roberta_embedding(
                    &device,
                    repo_id,
                    options.cache_dir,
                )
                .await
                .map_err(EmbeddingError::load)?,
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
            resolve_batch_size(inputs.len(), batch_size, DEFAULT_BATCH_SIZE)
                .map_err(EmbeddingError::inference)?;

        let mut embeddings = Vec::with_capacity(inputs.len());
        for batch in inputs.chunks(batch_size) {
            let batch_embeddings = match &self.model {
                LoadedEmbeddingModel::Bert(model) => model
                    .encode(batch, prompt, &self.device)
                    .map_err(EmbeddingError::inference)?,
                LoadedEmbeddingModel::Mpnet(model) => model
                    .encode(batch, prompt, &self.device)
                    .map_err(EmbeddingError::inference)?,
                LoadedEmbeddingModel::XlmRoberta(model) => model
                    .encode(batch, prompt, &self.device)
                    .map_err(EmbeddingError::inference)?,
            };
            embeddings.extend(
                tensor2_to_rows_f32(
                    batch_embeddings,
                    "failed to read embedding output tensor",
                )
                .map_err(EmbeddingError::inference)?,
            );
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
            .map_err(EmbeddingError::inference)
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
