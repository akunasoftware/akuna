use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use hf_hub::{Repo, RepoType};

const HF_REPO_TEST_CORPUS: &str = "akunasoftware/test-corpus";
const HF_REPO_TEST_CORPUS_REVISION: &str =
    "628188f924994038784e21e828518441634948ba";
const HF_REPO_CONTENT_PREFIX: &str = "content/fixtures";
static CORPUS_FIXTURES: OnceLock<Mutex<HashMap<String, PathBuf>>> =
    OnceLock::new();

/// Returns a named fixture from the pinned test corpus revision.
pub(crate) fn corpus_fixture(name: &str) -> Result<PathBuf> {
    let cache = CORPUS_FIXTURES.get_or_init(Default::default);
    let cached = {
        let cache = cache
            .lock()
            .map_err(|_| anyhow::anyhow!("corpus fixture cache poisoned"))?;
        cache.get(name).cloned()
    };
    if let Some(path) = cached {
        return Ok(path);
    }

    let client = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(false)
        .build()?;
    let repo = client.repo(Repo::with_revision(
        HF_REPO_TEST_CORPUS.to_string(),
        RepoType::Dataset,
        HF_REPO_TEST_CORPUS_REVISION.to_string(),
    ));
    let path = repo.download(&format!("{HF_REPO_CONTENT_PREFIX}/{name}"))?;
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("corpus fixture cache poisoned"))?;
    cache.insert(name.to_string(), path.clone());
    Ok(path)
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
