//! Native (pure-burn) PP-OCRv6 detection and recognition models that load
//! directly from HuggingFace safetensors — no ONNX-generated code or `.bpk`.

pub(in crate::ocr) mod det;
pub(in crate::ocr) mod det_medium;
pub(in crate::ocr) mod rec;
pub(in crate::ocr) mod rec_tiny;

mod lcnet;

use anyhow::{Context, Result};
use hf_hub::api::sync::Api;

/// Downloads `model.safetensors` from `repo` and returns its bytes.
pub(in crate::ocr) fn fetch_safetensors(repo: &str) -> Result<Vec<u8>> {
    let path = Api::new()
        .with_context(|| format!("failed to init Hugging Face API for {repo}"))?
        .model(repo.to_string())
        .get("model.safetensors")
        .with_context(|| format!("failed to fetch safetensors from {repo}"))?;
    std::fs::read(&path).with_context(|| {
        format!("failed to read safetensors at {}", path.display())
    })
}
