use anyhow::{Result, bail};

use crate::ocr::{
    OcrRecognitionModel, models::pp_ocr::spec::recognizer_dictionary,
};

pub(crate) fn load_dictionary(
    model: OcrRecognitionModel,
) -> Result<Vec<String>> {
    let mut dictionary = parse_character_dict(recognizer_dictionary(model))?;
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
