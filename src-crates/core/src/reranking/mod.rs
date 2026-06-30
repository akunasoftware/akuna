//! Cross-encoder text reranking.
//!
//! Scores and ranks documents against a query. Select a checkpoint via
//! `RerankingModel` (defaults to `BgeRerankerBase`). Inference is blocking;
//! wrap it in `tokio::task::spawn_blocking` from async contexts.
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
//!     // Note: `score` is blocking. In production, wrap heavy inference in
//!     // `tokio::task::spawn_blocking` to avoid stalling async workers.
//!     let score = model.score("Rust ML", "Burn is a Rust ML framework")?;
//!     assert!(score.is_finite());
//!     Ok(())
//! }
//! ```

mod models;

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use burn_dispatch::DispatchDevice;

use crate::ml::backend::{self, Backend};
use crate::ml::{resolve_batch_size, sigmoid_f32, tensor1_to_vec_f32};

use crate::reranking::models::xlm_roberta::{
    XlmRobertaRerankerModel, load_pretrained_xlm_roberta_reranker,
};
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
        let model = load_pretrained_xlm_roberta_reranker(
            &device,
            options.model.repo_id(),
            options.cache_dir,
        )
        .await?;

        Ok(Self { model, device })
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
            resolve_batch_size(pairs.len(), batch_size, DEFAULT_BATCH_SIZE)?;
        let mut scores = Vec::with_capacity(pairs.len());

        for batch in pairs.chunks(batch_size) {
            let batch_scores = self.model.score(batch, &self.device)?;
            scores.extend(tensor1_to_vec_f32(
                batch_scores,
                "failed to read reranker output tensor",
            )?);
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
        validate_top_k(options.top_k)?;
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

        indexed_scores.sort_by(|left, right| right.1.total_cmp(&left.1));
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

fn validate_top_k(top_k: Option<usize>) -> Result<()> {
    if matches!(top_k, Some(0)) {
        bail!("top_k must be greater than zero");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn util_top_k_validate_rejects_zero() {
        let error =
            validate_top_k(Some(0)).expect_err("zero top_k should fail");
        assert!(
            error
                .to_string()
                .contains("top_k must be greater than zero")
        );
    }

    #[test]
    fn util_top_k_validate_accepts_none_and_positive() {
        validate_top_k(None).expect("None top_k should pass");
        validate_top_k(Some(1)).expect("positive top_k should pass");
    }

    #[test]
    fn util_sigmoid_maps_scores_to_zero_one() {
        assert_eq!(sigmoid_f32(0.0), 0.5);
        assert!(sigmoid_f32(10.0) > 0.99);
        assert!(sigmoid_f32(-10.0) < 0.01);
    }

    #[test]
    fn util_sigmoid_bounded_for_extreme_scores() {
        assert!(sigmoid_f32(1000.0).is_finite());
        assert!(sigmoid_f32(-1000.0).is_finite());
        assert!(sigmoid_f32(1000.0) <= 1.0);
        assert!(sigmoid_f32(-1000.0) >= 0.0);
    }
}
