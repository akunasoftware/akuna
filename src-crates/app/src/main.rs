//! Command-line interface for knowledge tools: extraction, schema generation,
//! and the local REST API server.
//!
//! # Example
//!
//! ```text
//! akuna --help
//! akuna extract ./notes.md --metadata --text
//! akuna serve
//! ```

mod api;
mod cli;
mod tracing;

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
