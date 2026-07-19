//! Stack-contained inference helpers.

use std::future::Future;

use tokio::runtime::Handle;

// One fixed stack is enough until stack usage differs by model.
const FFI_STACK_SIZE: usize = 128 * 1024 * 1024;

/// Runs sync work on a larger stack.
pub(crate) fn run<T, F>(f: F) -> Result<T, String>
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    // 128MB wrapper contains current stack-heavy layout/OCR inference.
    std::thread::scope(|scope| {
        let handle = std::thread::Builder::new()
            .stack_size(FFI_STACK_SIZE)
            .spawn_scoped(scope, f)
            .map_err(|error| {
                format!("failed to start FFI stack wrapper: {error}")
            })?;
        handle.join().map_err(|panic| {
            let message = panic
                .downcast_ref::<&str>()
                .copied()
                .map(str::to_string)
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic payload".to_string());
            format!("FFI stack wrapper panicked: {message}")
        })
    })
}

/// Runs async work on the current runtime from a larger stack.
pub(crate) fn run_async<T>(
    future: impl Future<Output = T> + Send,
) -> Result<T, String>
where
    T: Send,
{
    let handle = Handle::current();
    run(|| handle.block_on(future))
}
