//! Shared ML model utilities.

#[cfg(feature = "ocr")]
use std::path::{Path, PathBuf};

#[cfg(feature = "ocr")]
use anyhow::{Context, Result};
#[cfg(any(feature = "embedding", feature = "ocr", feature = "reranking"))]
use burn::tensor::{Tensor, backend::Backend};
#[cfg(feature = "ocr")]
use hf_hub::api::tokio::ApiBuilder;
#[cfg(feature = "ocr")]
use hf_hub::{Repo, RepoType};

/// Shared ML backend selection.
pub(crate) mod backend;

/// Shared neural network layers.
#[cfg(feature = "ocr")]
pub(crate) mod burn_nn;

/// Image preprocessing helpers.
#[cfg(feature = "ocr")]
pub(crate) mod imageproc;

#[cfg(any(feature = "embedding", feature = "reranking"))]
pub(crate) mod text;

/// A BERT-style transformer encoder used by embedding and reranking.
#[cfg(any(feature = "embedding", feature = "reranking"))]
pub(crate) mod transformer;

#[cfg(test)]
mod tests;

/// Contraction-dim chunk size for [`safe_matmul`].
#[cfg(any(feature = "ocr", feature = "embedding", feature = "reranking"))]
pub(crate) const SAFE_MATMUL_K: usize = 256;

/// A pinned Hugging Face model weight.
#[cfg(feature = "ocr")]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HfWeight {
    pub(crate) repo_id: &'static str,
    pub(crate) revision: &'static str,
    pub(crate) filename: &'static str,
}

/// Preserves an internal model error at a public capability boundary.
#[cfg(any(feature = "embedding", feature = "ocr", feature = "reranking"))]
pub(crate) fn boxed_model_error(
    error: anyhow::Error,
) -> Box<dyn std::error::Error + Send + Sync> {
    error.into_boxed_dyn_error()
}

/// Fetches one pinned model weight into the configured cache.
#[cfg(feature = "ocr")]
pub(crate) async fn fetch_hf_weight(
    weight: HfWeight,
    cache_dir: Option<&Path>,
    context: &str,
) -> Result<PathBuf> {
    let mut builder = ApiBuilder::new().with_progress(true);
    if let Some(cache_dir) = cache_dir {
        builder = builder.with_cache_dir(cache_dir.to_path_buf());
    }

    let api = builder.build().with_context(|| {
        format!("failed to initialize Hugging Face API for {context}")
    })?;
    let repo = api.repo(Repo::with_revision(
        weight.repo_id.to_string(),
        RepoType::Model,
        weight.revision.to_string(),
    ));

    repo.get(weight.filename).await.with_context(|| {
        format!(
            "failed to fetch {context} weight {} from {} at {}",
            weight.filename, weight.repo_id, weight.revision
        )
    })
}

/// Computes `lhs @ rhs` for tensors of any rank.
#[cfg(any(feature = "ocr", feature = "embedding", feature = "reranking"))]
pub(crate) fn safe_matmul<B: Backend, const D: usize>(
    lhs: Tensor<B, D>,
    rhs: Tensor<B, D>,
) -> Tensor<B, D> {
    let k = lhs.dims()[D - 1];
    if k <= SAFE_MATMUL_K {
        return lhs.matmul(rhs);
    }
    let mut acc: Option<Tensor<B, D>> = None;
    let mut start = 0usize;
    while start < k {
        let len = (k - start).min(SAFE_MATMUL_K);
        let part = lhs
            .clone()
            .narrow(D - 1, start, len)
            .matmul(rhs.clone().narrow(D - 2, start, len));
        acc = Some(match acc {
            None => part,
            Some(previous) => previous + part,
        });
        start += len;
    }
    acc.expect("safe_matmul K > 0")
}

/// Computes numerically simple sigmoid for model scores.
#[cfg(any(feature = "ocr", feature = "reranking"))]
pub(crate) fn sigmoid_f32(score: f32) -> f32 {
    1.0 / (1.0 + (-score).exp())
}
