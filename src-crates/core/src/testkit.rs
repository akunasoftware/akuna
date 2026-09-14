use anyhow::Result;

#[cfg(any(feature = "extraction", feature = "ocr"))]
pub(crate) use corpus::corpus_fixture;

#[cfg(any(feature = "extraction", feature = "ocr"))]
mod corpus {
    use std::path::PathBuf;

    use anyhow::Result;
    use hf_hub::{Repo, RepoType};

    const HF_REPO_TEST_CORPUS: &str = "akunasoftware/test-corpus";
    const HF_REPO_TEST_CORPUS_REVISION: &str =
        "2c8f0c235151a8a241ef4568b1ecab6c95fbb3f1";
    const HF_REPO_CONTENT_PREFIX: &str = "content/fixtures";

    /// Returns a named fixture from the pinned test corpus revision.
    pub(crate) fn corpus_fixture(name: &str) -> Result<PathBuf> {
        let client = hf_hub::api::sync::ApiBuilder::new()
            .with_progress(false)
            .build()?;
        let repo = client.repo(Repo::with_revision(
            HF_REPO_TEST_CORPUS.to_string(),
            RepoType::Dataset,
            HF_REPO_TEST_CORPUS_REVISION.to_string(),
        ));
        Ok(repo.get(&format!("{HF_REPO_CONTENT_PREFIX}/{name}"))?)
    }
}

/// Runs model-heavy tests on a larger stack.
pub(crate) fn run_with_model_stack<F>(f: F) -> Result<()>
where
    F: FnOnce() -> Result<()> + Send + 'static,
{
    let handle = anyhow::Context::context(
        std::thread::Builder::new()
            .stack_size(128 * 1024 * 1024)
            .spawn(f),
        "failed to spawn model test thread",
    )?;
    handle.join().map_err(|panic| {
        let message = panic
            .downcast_ref::<&str>()
            .copied()
            .map(str::to_string)
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "model test thread panicked".to_string());
        anyhow::anyhow!(message)
    })?
}
