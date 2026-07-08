//! Shared ML model utilities.

#[cfg(any(feature = "layout", feature = "ocr"))]
use std::path::Path;
#[cfg(any(
    feature = "embedding",
    feature = "layout",
    feature = "ocr",
    feature = "reranking"
))]
use std::path::PathBuf;

#[cfg(any(feature = "embedding", feature = "reranking"))]
use anyhow::bail;
#[cfg(any(
    feature = "embedding",
    feature = "layout",
    feature = "ocr",
    feature = "reranking"
))]
use anyhow::{Context, Result};
#[cfg(any(
    feature = "embedding",
    feature = "layout",
    feature = "ocr",
    feature = "reranking"
))]
use burn::tensor::{Tensor, backend::Backend};
#[cfg(any(
    feature = "embedding",
    feature = "layout",
    feature = "ocr",
    feature = "reranking"
))]
use hf_hub::api::tokio::ApiBuilder;
#[cfg(any(feature = "layout", feature = "ocr"))]
use hf_hub::{Repo, RepoType};

/// Shared ML backend selection.
pub(crate) mod backend;

/// Shared neural network layers.
#[cfg(feature = "layout")]
pub(crate) mod burn_nn;

/// Image preprocessing helpers.
#[cfg(any(feature = "layout", feature = "ocr"))]
pub(crate) mod imageproc;

/// A BERT-style transformer encoder used by embedding and reranking.
#[cfg(any(feature = "embedding", feature = "reranking"))]
pub(crate) mod transformer;

#[cfg(test)]
mod tests;

/// Contraction-dim chunk size for [`safe_matmul`].
#[cfg(any(feature = "layout", feature = "embedding", feature = "reranking"))]
pub(crate) const SAFE_MATMUL_K: usize = 256;

/// A pinned Hugging Face model weight.
#[cfg(any(feature = "layout", feature = "ocr"))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HfWeight {
    pub(crate) repo_id: &'static str,
    pub(crate) revision: &'static str,
    pub(crate) filename: &'static str,
}

/// Preserves an internal model error at a public capability boundary.
#[cfg(any(
    feature = "embedding",
    feature = "layout",
    feature = "ocr",
    feature = "reranking"
))]
pub(crate) fn boxed_model_error(
    error: anyhow::Error,
) -> Box<dyn std::error::Error + Send + Sync> {
    error.into_boxed_dyn_error()
}

/// Fetches one pinned model weight into the configured cache.
#[cfg(any(feature = "layout", feature = "ocr"))]
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
#[cfg(any(feature = "layout", feature = "embedding", feature = "reranking"))]
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

/// Resolved model asset paths.
#[cfg(any(feature = "embedding", feature = "reranking"))]
#[derive(Debug, Clone)]
pub(crate) struct HfModelFiles {
    pub(crate) config_path: PathBuf,
    pub(crate) weights_path: PathBuf,
    pub(crate) tokenizer_path: PathBuf,
}

/// Resolves required model assets.
#[cfg(any(feature = "embedding", feature = "reranking"))]
pub(crate) async fn download_hf_model_files(
    repo_id: &str,
    weights_file: &str,
    cache_dir: Option<PathBuf>,
    context: &str,
) -> Result<HfModelFiles> {
    let mut builder = ApiBuilder::new().with_progress(true);
    if let Some(cache_dir) = cache_dir {
        builder = builder.with_cache_dir(cache_dir);
    }

    let api = builder.build().with_context(|| {
        format!("failed to initialize Hugging Face API for {context}")
    })?;
    let repo = api.model(repo_id.to_string());

    let config_path = repo.get("config.json").await.with_context(|| {
        format!("failed to fetch {context} config for {repo_id}")
    })?;
    let weights_path = repo.get(weights_file).await.with_context(|| {
        format!("failed to fetch {context} weights for {repo_id}")
    })?;
    let tokenizer_path =
        repo.get("tokenizer.json").await.with_context(|| {
            format!("failed to fetch {context} tokenizer for {repo_id}")
        })?;
    Ok(HfModelFiles {
        config_path,
        weights_path,
        tokenizer_path,
    })
}

/// Loads a JSON model config.
#[cfg(any(feature = "embedding", feature = "reranking"))]
pub(crate) fn load_json_config<T: serde::de::DeserializeOwned>(
    path: &std::path::Path,
    what: &str,
) -> Result<T> {
    let content = std::fs::read_to_string(path).with_context(|| {
        format!("failed to read {what} at {}", path.display())
    })?;
    serde_json::from_str(&content).with_context(|| {
        format!("failed to parse {what} at {}", path.display())
    })
}

/// Resolves an optional batch size, applying a default and validation.
#[cfg(any(feature = "embedding", feature = "reranking"))]
pub(crate) fn resolve_batch_size(
    item_count: usize,
    requested: Option<usize>,
    default_limit: usize,
) -> Result<usize> {
    let batch_size = requested.unwrap_or(item_count.min(default_limit));
    if batch_size == 0 {
        bail!("batch size must be greater than zero");
    }

    Ok(batch_size)
}

/// Builds XLM-RoBERTa position ids from padded attention masks.
#[cfg(any(feature = "embedding", feature = "reranking"))]
pub(crate) fn xlm_roberta_position_ids(
    attention_mask: &[f32],
    batch_size: usize,
    sequence_length: usize,
    pad_token_id: i32,
) -> Vec<i32> {
    let mut position_ids = vec![pad_token_id; attention_mask.len()];
    for batch in 0..batch_size {
        let mut position = pad_token_id + 1;
        for offset in batch * sequence_length..(batch + 1) * sequence_length {
            if attention_mask[offset] == 0.0 {
                continue;
            }
            position_ids[offset] = position;
            position += 1;
        }
    }

    position_ids
}

/// Converts a rank-1 tensor to `Vec<f32>`.
#[cfg(feature = "reranking")]
pub(crate) fn tensor1_to_vec_f32<B: Backend>(
    tensor: Tensor<B, 1>,
    context: &str,
) -> Result<Vec<f32>> {
    let data = tensor.into_data().convert::<f32>();
    data.as_slice::<f32>()
        .map(|values| values.to_vec())
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context(context.to_string())
}

/// Converts a rank-2 tensor to `Vec<Vec<f32>>` rows.
#[cfg(feature = "embedding")]
pub(crate) fn tensor2_to_rows_f32<B: Backend>(
    tensor: Tensor<B, 2>,
    context: &str,
) -> Result<Vec<Vec<f32>>> {
    let [row_count, column_count] = tensor.dims();
    let data = tensor.into_data().convert::<f32>();
    let values = data
        .as_slice::<f32>()
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context(context.to_string())?;

    Ok(values
        .chunks(column_count)
        .take(row_count)
        .map(|row| row.to_vec())
        .collect())
}

/// Computes numerically simple sigmoid for model scores.
#[cfg(any(feature = "layout", feature = "reranking"))]
pub(crate) fn sigmoid_f32(score: f32) -> f32 {
    1.0 / (1.0 + (-score).exp())
}

/// Selects the first token embedding from a sequence output.
#[cfg(any(feature = "embedding", feature = "reranking"))]
pub(crate) fn cls_pooling<B: Backend>(
    hidden_states: Tensor<B, 3>,
) -> Tensor<B, 2> {
    let [batch_size, seq_len, hidden_size] = hidden_states.dims();
    let device = hidden_states.device();
    let mut mask = vec![0.0f32; batch_size * seq_len];
    for batch_index in 0..batch_size {
        mask[batch_index * seq_len] = 1.0;
    }

    let mask = Tensor::<B, 1>::from_floats(mask.as_slice(), &device)
        .reshape([batch_size, seq_len, 1])
        .expand([batch_size, seq_len, hidden_size]);

    (hidden_states * mask)
        .sum_dim(1)
        .reshape([batch_size, hidden_size])
}
