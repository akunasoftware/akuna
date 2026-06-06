//! Command-line interface for knowledge tools.
//!
//! Wires together extraction, schema generation, and the local REST API
//! server on top of the shared `akuna-core` workspace crate.
//!
//! Subcommands:
//! - `extract`: extract metadata and structured content parts from a file
//! - `schemas`: generate app and OpenAPI JSON schemas
//! - `serve`: run the local HTTP API server
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
mod config;
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
