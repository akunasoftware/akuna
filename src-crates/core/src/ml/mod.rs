//! Shared ML model utilities.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use burn::tensor::{Tensor, backend::Backend};
use hf_hub::api::tokio::ApiBuilder;

/// Runtime backend and device selection, shared by every ML module.
pub(crate) mod backend;

/// Reusable native-burn layers loaded from safetensors.
#[cfg(feature = "layout")]
pub(crate) mod burn_nn;

/// cv2-compatible image preprocessing.
#[cfg(any(feature = "layout", feature = "ocr"))]
pub(crate) mod imageproc;

/// A BERT-style transformer encoder used by embedding and reranking.
#[cfg(any(feature = "embedding", feature = "reranking"))]
pub(crate) mod transformer;

/// Contraction-dim chunk size for [`safe_matmul`].
pub(crate) const SAFE_MATMUL_K: usize = 256;

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

/// Local filesystem paths to common Hugging Face model files.
#[derive(Debug, Clone)]
pub(crate) struct HfModelFiles {
    pub(crate) config_path: PathBuf,
    pub(crate) weights_path: PathBuf,
    pub(crate) tokenizer_path: PathBuf,
    pub(crate) sentence_bert_config_path: Option<PathBuf>,
}

/// Downloads common Hugging Face model files into the local cache.
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
    let sentence_bert_config_path =
        repo.get("sentence_bert_config.json").await.ok();

    Ok(HfModelFiles {
        config_path,
        weights_path,
        tokenizer_path,
        sentence_bert_config_path,
    })
}

/// Reads and deserializes a JSON config file, tagging errors with `what`.
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

/// Converts a rank-1 tensor to `Vec<f32>`.
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
pub(crate) fn sigmoid_f32(score: f32) -> f32 {
    1.0 / (1.0 + (-score).exp())
}

/// Selects the first token embedding from a sequence output.
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
