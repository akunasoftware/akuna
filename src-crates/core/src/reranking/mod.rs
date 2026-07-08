//! Cross-encoder text reranking.
//!
//! Scores and ranks documents against a query.
//!
//! # Example
//!
//! ```rust,no_run
//! use akuna_core::reranking::{RerankingModel, TextReranker, TextRerankerOptions};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let model = TextReranker::new(TextRerankerOptions {
//!         model: RerankingModel::BgeRerankerBase,
//!         ..Default::default()
//!     })
//!     .await?;
//!     let score = model.score("Rust ML", "Burn is a Rust ML framework")?;
//!     assert!(score.is_finite());
//!     Ok(())
//! }
//! ```

mod error;
mod models;

#[cfg(test)]
mod tests;

use std::path::PathBuf;

use anyhow::{Context, bail};
use burn_dispatch::DispatchDevice;

use crate::ml::backend::{self, Backend};
use crate::ml::{resolve_batch_size, sigmoid_f32, tensor1_to_vec_f32};

use crate::reranking::models::xlm_roberta::{
    XlmRobertaRerankerModel, load_pretrained_xlm_roberta_reranker,
};

pub use error::RerankingError;

type Result<T> = std::result::Result<T, RerankingError>;
/// Default inference batch size when callers do not override it.
const DEFAULT_BATCH_SIZE: usize = 32;

/// Supported reranker model checkpoints.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RerankingModel {
    /// `BAAI/bge-reranker-base`.
    #[default]
    BgeRerankerBase,
}

impl RerankingModel {
    /// Hugging Face repository for this checkpoint.
    fn repo_id(self) -> &'static str {
        match self {
            Self::BgeRerankerBase => "BAAI/bge-reranker-base",
        }
    }
}

/// Construction options for a [`TextReranker`].
#[derive(Debug, Clone, Default)]
pub struct TextRerankerOptions {
    /// Which reranker checkpoint to load.
    pub model: RerankingModel,
    /// Optional Hugging Face cache directory override.
    pub cache_dir: Option<PathBuf>,
}

/// A single document ranked against a query.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct RerankResult {
    /// Original index of the document in the input slice.
    pub index: usize,
    /// Relevance score for the document.
    pub score: f32,
    /// Document text the score applies to.
    pub document: String,
}

/// Tunable behaviour for a single [`TextReranker::rerank_with_options`] call.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RerankOptions {
    /// Keep only the top `n` results when set.
    pub top_k: Option<usize>,
    /// Apply sigmoid normalization to scores.
    pub normalize: bool,
    /// Override the default inference batch size.
    pub batch_size: Option<usize>,
}

/// Cross-encoder text reranker.
pub struct TextReranker {
    model: XlmRobertaRerankerModel<Backend>,
    model_kind: RerankingModel,
    device: DispatchDevice,
}

impl TextReranker {
    /// Loads a reranker from `options` onto the default device.
    pub async fn new(options: TextRerankerOptions) -> Result<Self> {
        Self::new_on(backend::active_device(), options).await
    }

    /// Loads a reranker from `options` onto a specific device.
    pub(crate) async fn new_on(
        device: DispatchDevice,
        options: TextRerankerOptions,
    ) -> Result<Self> {
        let model_kind = options.model;
        let model = load_pretrained_xlm_roberta_reranker(
            &device,
            model_kind.repo_id(),
            options.cache_dir,
        )
        .await
        .map_err(RerankingError::load)?;

        Ok(Self {
            model,
            model_kind,
            device,
        })
    }

    /// Returns the loaded reranking checkpoint.
    pub fn model(&self) -> RerankingModel {
        self.model_kind
    }

    /// Scores query/document pairs in batches, one score per pair.
    fn score_batch_inner(
        &self,
        pairs: &[(&str, &str)],
        batch_size: Option<usize>,
    ) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size =
            resolve_batch_size(pairs.len(), batch_size, DEFAULT_BATCH_SIZE)
                .map_err(RerankingError::inference)?;
        let mut scores = Vec::with_capacity(pairs.len());

        for batch in pairs.chunks(batch_size) {
            let batch_scores = self
                .model
                .score(batch, &self.device)
                .map_err(RerankingError::inference)?;
            scores.extend(
                tensor1_to_vec_f32(
                    batch_scores,
                    "failed to read reranker output tensor",
                )
                .map_err(RerankingError::inference)?,
            );
        }

        Ok(scores)
    }

    /// Scores a single query/document pair.
    pub fn score(
        &self,
        query: impl AsRef<str>,
        document: impl AsRef<str>,
    ) -> Result<f32> {
        let mut scores = self
            .score_batch_inner(&[(query.as_ref(), document.as_ref())], None)?;
        scores
            .pop()
            .context("expected one score for a single input pair")
            .map_err(RerankingError::inference)
    }

    /// Scores many query/document pairs in batches.
    pub fn score_batch<Q, D>(
        &self,
        pairs: &[(Q, D)],
        batch_size: Option<usize>,
    ) -> Result<Vec<f32>>
    where
        Q: AsRef<str>,
        D: AsRef<str>,
    {
        let refs = pairs
            .iter()
            .map(|(query, document)| (query.as_ref(), document.as_ref()))
            .collect::<Vec<_>>();
        self.score_batch_inner(&refs, batch_size)
    }

    /// Ranks documents against a query using default options.
    pub fn rerank<S: AsRef<str>>(
        &self,
        query: impl AsRef<str>,
        documents: &[S],
    ) -> Result<Vec<RerankResult>> {
        self.rerank_with_options(query, documents, Default::default())
    }

    /// Ranks documents against a query with custom options.
    pub fn rerank_with_options<S: AsRef<str>>(
        &self,
        query: impl AsRef<str>,
        documents: &[S],
        options: RerankOptions,
    ) -> Result<Vec<RerankResult>> {
        validate_top_k(options.top_k).map_err(RerankingError::inference)?;
        let query = query.as_ref();
        let document_refs =
            documents.iter().map(AsRef::as_ref).collect::<Vec<_>>();
        let pairs = document_refs
            .iter()
            .map(|document| (query, *document))
            .collect::<Vec<_>>();
        let scores = self.score_batch_inner(&pairs, options.batch_size)?;
        let mut indexed_scores = scores
            .into_iter()
            .enumerate()
            .map(|(index, score)| {
                let score = if options.normalize {
                    sigmoid_f32(score)
                } else {
                    score
                };
                (index, score)
            })
            .collect::<Vec<_>>();

        indexed_scores.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        if let Some(top_k) = options.top_k {
            indexed_scores.truncate(top_k);
        }

        Ok(indexed_scores
            .into_iter()
            .map(|(index, score)| RerankResult {
                index,
                score,
                document: document_refs[index].to_string(),
            })
            .collect())
    }
}

fn validate_top_k(top_k: Option<usize>) -> anyhow::Result<()> {
    if matches!(top_k, Some(0)) {
        bail!("top_k must be greater than zero");
    }
    Ok(())
}
