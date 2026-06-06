//! Application configuration loading and persistence.

use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const CONFIG_SCHEMA_FILE_NAME: &str = "config.json";

/// Root app configuration persisted as YAML.
#[derive(Debug, Clone, Default, Deserialize, JsonSchema, Serialize)]
pub(crate) struct AppConfig {}

impl AppConfig {
    /// Generates and writes JSON schema for app config.
    pub(crate) fn generate_schema(export_path: PathBuf) -> Result<PathBuf> {
        let schema = schemars::schema_for!(AppConfig);
        let schema_json = serde_json::to_string_pretty(&schema)
            .context("Failed to serialize config schema")?;

        fs::write(&export_path, schema_json).with_context(|| {
            format!("Failed to write {}", export_path.display())
        })?;

        Ok(export_path)
    }
}

/// Returns config JSON schema file name.
pub(crate) fn config_schema_file_name() -> &'static str {
    CONFIG_SCHEMA_FILE_NAME
}
