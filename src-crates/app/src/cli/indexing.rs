//! Indexing pipeline command.

use std::{
    collections::HashMap, path::PathBuf, thread::available_parallelism,
    time::Instant,
};

use akuna_core::extraction::{ExtractionConfig, extract_file};
use anyhow::{Context, Result, ensure};
use clap::Args;
use ignore::WalkBuilder;
use tokio::{runtime::Handle, task::JoinSet};

/// CLI arguments for the `index` command.
#[derive(Args)]
pub(crate) struct IndexCommand {
    /// Path to the directory to index.
    directory: PathBuf,
}

impl IndexCommand {
    /// Indexes the contents of a directory for search.
    pub(crate) async fn run(self) -> Result<()> {
        let started = Instant::now();
        let scan_root = self.directory.canonicalize().with_context(|| {
            format!("resolving {}", self.directory.display())
        })?;
        ensure!(
            scan_root.is_dir(),
            "not a directory: {}",
            scan_root.display()
        );
        let mut files = Vec::new();
        for entry in WalkBuilder::new(&scan_root).build() {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    tracing::error!("walking {}: {error}", scan_root.display());
                    continue;
                }
            };
            if let Some(error) = entry.error() {
                tracing::error!("walking {}: {error}", entry.path().display());
                continue;
            }
            if !entry.file_type().is_some_and(|kind| kind.is_file()) {
                continue;
            }
            files.push(entry.into_path());
        }
        files.sort();

        // Extract content only; persistence arrives with the real indexer.
        let concurrency =
            available_parallelism().map_or(1, |count| count.get().min(64));
        let runtime = Handle::current();
        let mut remaining = files.iter();
        let mut jobs = JoinSet::new();
        let mut paths = HashMap::new();
        let mut successes = 0;
        let mut failed = 0;

        loop {
            for path in remaining.by_ref().take(concurrency - jobs.len()) {
                let source = path.clone();
                let runtime = runtime.clone();
                let job = jobs.spawn_blocking(move || {
                    runtime.block_on(async {
                        extract_file(
                            &source,
                            &ExtractionConfig {
                                return_content: true,
                                ..Default::default()
                            },
                        )
                        .await
                        .map_err(anyhow::Error::from)
                    })
                });
                paths.insert(job.id(), path);
            }

            let Some(completed) = jobs.join_next_with_id().await else {
                break;
            };
            let (id, result) = match completed {
                Ok((id, result)) => (id, result),
                Err(error) => (error.id(), Err(error.into())),
            };
            let path = paths
                .remove(&id)
                .expect("every admitted job has a source path");
            match result {
                Ok(_) => successes += 1,
                Err(error) => {
                    failed += 1;
                    tracing::error!("extracting {}: {error:#}", path.display());
                }
            }
        }

        println!(
            "Files (ignore rules applied): {} successes: {successes} failed: {failed} seconds: {:.3} nothing persisted.",
            files.len(),
            started.elapsed().as_secs_f64()
        );

        Ok(())
    }
}
