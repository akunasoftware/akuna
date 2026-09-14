//! Command-line interface commands.

mod extraction;
mod indexing;

use crate::tracing::{LOG_LEVELS, setup_tracing};
use anyhow::Result;
use clap::{Parser, Subcommand};

/// Parsed top-level CLI input.
#[derive(Parser)]
#[command(version, about = "Command-line tools")]
struct Cli {
    #[arg(long, value_parser = LOG_LEVELS)]
    log_level: Option<String>,

    #[command(subcommand)]
    command: Command,
}

/// Available CLI subcommands.
#[derive(Subcommand)]
enum Command {
    /// Extract structured metadata & content from a file.
    Extract(extraction::ExtractCommand),

    /// Index contents of a directory for search.
    Index(indexing::IndexCommand),
}

/// Runs the parsed CLI command.
pub async fn run() -> Result<()> {
    let cli = Cli::parse();
    setup_tracing(crate::APP_NAME, cli.log_level.as_deref());

    match cli.command {
        Command::Extract(command) => command.run().await,
        Command::Index(command) => command.run().await,
    }
}
