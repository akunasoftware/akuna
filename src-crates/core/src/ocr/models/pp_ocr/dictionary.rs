use std::path::Path;

use anyhow::{Context, Result, bail};
use hf_hub::{Repo, RepoType, api::tokio::ApiBuilder};

use crate::ocr::models::pp_ocr::spec::PpOcrRecognitionConfig;

const INFERENCE_CONFIG_FILE: &str = "inference.yml";

pub(crate) async fn load_dictionary(
    config: &PpOcrRecognitionConfig,
    cache_dir: Option<&Path>,
) -> Result<Vec<String>> {
    let mut builder = ApiBuilder::new().with_progress(true);
    if let Some(cache_dir) = cache_dir {
        builder = builder.with_cache_dir(cache_dir.to_path_buf());
    }

    let api = builder.build().context(
        "failed to initialize Hugging Face API for PaddleOCR dictionary",
    )?;
    let repo = api.repo(Repo::with_revision(
        config.spec.repo_id.to_string(),
        RepoType::Model,
        config.spec.revision.to_string(),
    ));

    let inference_config_path = repo.get(INFERENCE_CONFIG_FILE).await.with_context(|| {
        format!(
            "failed to fetch PaddleOCR recognizer {INFERENCE_CONFIG_FILE} from {}",
            config.spec.repo_id
        )
    })?;
    let inference_config = std::fs::read_to_string(&inference_config_path)
        .with_context(|| {
            format!("failed to read {}", inference_config_path.display())
        })?;

    let mut dictionary = parse_character_dict(&inference_config)?;
    dictionary.push(" ".to_string());

    Ok(dictionary)
}

fn parse_character_dict(yaml: &str) -> Result<Vec<String>> {
    let mut in_postprocess = false;
    let mut postprocess_indent = 0;
    let mut character_dict_indent = 0;
    let mut in_character_dict = false;
    let mut dictionary = Vec::new();

    for line in yaml.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let indent = line.len() - line.trim_start().len();
        if in_character_dict {
            if indent <= character_dict_indent && !trimmed.starts_with('-') {
                break;
            }

            let Some(value) = trimmed.strip_prefix('-') else {
                continue;
            };
            dictionary.push(parse_yaml_scalar(value.trim())?);
            continue;
        }

        if in_postprocess && indent <= postprocess_indent {
            in_postprocess = false;
        }

        if !in_postprocess {
            if trimmed == "PostProcess:" {
                in_postprocess = true;
                postprocess_indent = indent;
            }
            continue;
        }

        if let Some(value) = trimmed.strip_prefix("character_dict:") {
            character_dict_indent = indent;
            let value = value.trim();
            if value.is_empty() {
                in_character_dict = true;
                continue;
            }

            dictionary.extend(parse_inline_yaml_list(value)?);
            break;
        }
    }

    if dictionary.is_empty() {
        bail!(
            "PaddleOCR recognizer inference.yml missing PostProcess.character_dict"
        )
    }

    Ok(dictionary)
}

fn parse_inline_yaml_list(value: &str) -> Result<Vec<String>> {
    let Some(value) = value.strip_prefix('[').and_then(|v| v.strip_suffix(']'))
    else {
        bail!("unsupported inline character_dict YAML value")
    };

    value
        .split(',')
        .map(|item| parse_yaml_scalar(item.trim()))
        .collect()
}

fn parse_yaml_scalar(value: &str) -> Result<String> {
    if let Some(value) =
        value.strip_prefix('"').and_then(|v| v.strip_suffix('"'))
    {
        return Ok(value.replace("\\\"", "\"").replace("\\\\", "\\"));
    }

    if let Some(value) =
        value.strip_prefix('\'').and_then(|v| v.strip_suffix('\''))
    {
        return Ok(value.replace("''", "'"));
    }

    Ok(value
        .split_once(" #")
        .map_or(value, |(value, _)| value)
        .to_string())
}
