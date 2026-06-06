//! API server CLI command.

use anyhow::Result;

/// Starts the local API server.
pub async fn run() -> Result<()> {
    crate::api::server::run().await
}
