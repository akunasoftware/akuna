//! Command-line interface for knowledge tools.
//!
//! # Example
//!
//! ```text
//! <app> --help
//! <app> extract ./notes.md --metadata --text
//! ```

mod cli;
mod tracing;

pub(crate) use akuna_core::PACKAGE_NAME as APP_NAME;
use anyhow::Result;
use serde::Serialize;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    cli::run().await
}

pub(crate) fn print_json(value: &impl Serialize) -> Result<()> {
    serde_json::to_writer_pretty(std::io::stdout(), value)?;
    println!();
    Ok(())
}
