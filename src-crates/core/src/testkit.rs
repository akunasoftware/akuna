use anyhow::Result;

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
