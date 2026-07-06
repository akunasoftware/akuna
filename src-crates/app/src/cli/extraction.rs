use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

use crate::print_json;

/// CLI arguments for the `extract` command.
#[derive(Args)]
pub(crate) struct ExtractCommand {
    /// Path to the file to extract.
    file: PathBuf,
    /// Include detected file metadata in the result.
    #[arg(long)]
    metadata: bool,
    /// Print extracted text.
    #[arg(long)]
    text: bool,
    /// Include structured content parts (with source provenance) in the result.
    #[arg(long)]
    parts: bool,
}

impl ExtractCommand {
    /// Runs file extraction and prints the result.
    pub(crate) async fn run(self) -> Result<()> {
        tracing::info!("extracting data from {}", self.file.display());

        let full = !self.metadata && !self.text && !self.parts;

        let extraction = akuna_core::extraction::extract_file(
            &self.file,
            &akuna_core::extraction::ExtractionConfig {
                return_metadata: full || self.metadata,
                return_content: full || self.text,
                return_parts: full || self.parts,
                ..Default::default()
            },
        )
        .await?;

        if self.text && !self.metadata && !self.parts {
            let text = extraction.text.ok_or_else(|| {
                anyhow::anyhow!("no text content extracted for file")
            })?;
            print!("{text}");
            return Ok(());
        }

        print_json(&extraction)
    }
}
