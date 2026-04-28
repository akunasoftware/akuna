//! Command-line application entry point.

mod extraction;

use akuna_core::tracing::{LOG_LEVELS, setup_tracing};
use anyhow::Result;
use clap::{Parser, Subcommand};
use serde::Serialize;

#[derive(Parser)]
#[command(version, about = "Command-line tools")]
struct Cli {
    #[arg(long, value_parser = LOG_LEVELS)]
    log_level: Option<String>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Extract structured metadata & content from a file.
    Extract(extraction::ExtractCommand),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    setup_tracing(cli.log_level.as_deref());

    match cli.command {
        Command::Extract(command) => command.run().await?,
    }

    Ok(())
}

pub(crate) fn print_json(value: &impl Serialize) -> Result<()> {
    serde_json::to_writer_pretty(std::io::stdout(), value)?;
    println!();
    Ok(())
}
