use std::path::PathBuf;

use akuna_core::ak_info;
use anyhow::Result;
use clap::Args;

use crate::print_json;

#[derive(Args)]
pub(crate) struct ExtractCommand {
    file: PathBuf,
    /// Include all extracted data in the result.
    #[arg(long)]
    full: bool,
    /// Include extracted text content in the result.
    #[arg(long)]
    content: bool,
    /// Include extracted text chunks in the result.
    #[arg(long)]
    chunk: bool,
}

impl ExtractCommand {
    pub(crate) async fn run(self) -> Result<()> {
        ak_info!("extracting data from {}", self.file.display());

        let full = self.full || (!self.content && !self.chunk);

        let extraction = akuna_core::extraction::extract_file(
            self.file,
            &akuna_core::ExtractionConfig {
                return_metadata: true,
                return_content: full || self.content,
                return_chunking: full || self.chunk,
                text: Some(akuna_core::TextExtractionConfig::default()),
                chunking: Some(akuna_core::ChunkingConfig::default()),
            },
        )
        .await?;

        print_json(&extraction)
    }
}
