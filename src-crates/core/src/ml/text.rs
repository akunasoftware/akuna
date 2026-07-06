use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use burn::tensor::{Tensor, backend::Backend};
use hf_hub::api::tokio::ApiBuilder;

/// Resolved model asset paths.
#[derive(Debug, Clone)]
pub(crate) struct HfModelFiles {
    pub(crate) config_path: PathBuf,
    pub(crate) weights_path: PathBuf,
    pub(crate) tokenizer_path: PathBuf,
}

/// Resolves required model assets.
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
pub(crate) fn load_json_config<T: serde::de::DeserializeOwned>(
    path: &Path,
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

/// Validates that model inference returned one output per input.
pub(crate) fn validate_output_count(
    actor: &str,
    input_count: usize,
    output_count: usize,
) -> Result<()> {
    if input_count == output_count {
        return Ok(());
    }

    bail!(
        "{actor} output count mismatch: expected {input_count}, got {output_count}",
    )
}

/// Builds XLM-RoBERTa position ids from padded attention masks.
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
