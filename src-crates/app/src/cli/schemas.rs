//! Schema CLI commands.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Subcommand;

use crate::config::{AppConfig, config_schema_file_name};

/// Schema CLI commands.
#[derive(Subcommand)]
pub(crate) enum SchemasCommand {
    /// Generate app-owned schemas.
    Generate {
        /// Directory to write generated artifacts into.
        #[arg(short, long)]
        path: Option<PathBuf>,
    },
}

impl SchemasCommand {
    /// Runs selected schema command.
    pub(crate) async fn run(self) -> Result<()> {
        match self {
            Self::Generate { path: out } => generate_schemas(out),
        }
    }
}

/// Generates schema artifacts into `out_dir`, defaulting to the current
/// directory when none is given.
fn generate_schemas(out_dir: Option<PathBuf>) -> Result<()> {
    let out_dir = match out_dir {
        Some(out_dir) => out_dir,
        None => std::env::current_dir()
            .context("Failed to read current directory")?,
    };

    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("Failed to create {}", out_dir.display()))?;

    let config_schema_path =
        AppConfig::generate_schema(out_dir.join(config_schema_file_name()))?;
    let openapi_schema_path = crate::api::server::generate_schema(&out_dir)?;

    crate::print_json(&serde_json::json!({
        "config_schema": config_schema_path,
        "openapi_schema": openapi_schema_path,
    }))
}
