#[doc(hidden)]
pub use ::tracing::{debug, error, info, trace, warn};

use tracing_subscriber::EnvFilter;

const DEFAULT_LOG_LEVEL: &str = "warn";

/// Supported runtime log levels.
pub const LOG_LEVELS: [&str; 5] = ["trace", "debug", "info", "warn", "error"];

/// Initializes application tracing.
pub fn setup_tracing(log_level: Option<&str>) {
    let filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::from_default_env()
    } else {
        let level = log_level.unwrap_or(DEFAULT_LOG_LEVEL);
        EnvFilter::new(format!("{}={level}", crate::APP_NAME))
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}

/// Emits a trace-level event using the application target.
#[macro_export]
macro_rules! ak_trace {
    ($($arg:tt)+) => {
        $crate::tracing::trace!(target: $crate::APP_NAME, $($arg)+)
    };
}

/// Emits a debug-level event using the application target.
#[macro_export]
macro_rules! ak_debug {
    ($($arg:tt)+) => {
        $crate::tracing::debug!(target: $crate::APP_NAME, $($arg)+)
    };
}

/// Emits an info-level event using the application target.
#[macro_export]
macro_rules! ak_info {
    ($($arg:tt)+) => {
        $crate::tracing::info!(target: $crate::APP_NAME, $($arg)+)
    };
}

/// Emits a warn-level event using the application target.
#[macro_export]
macro_rules! ak_warn {
    ($($arg:tt)+) => {
        $crate::tracing::warn!(target: $crate::APP_NAME, $($arg)+)
    };
}

/// Emits an error-level event using the application target.
#[macro_export]
macro_rules! ak_error {
    ($($arg:tt)+) => {
        $crate::tracing::error!(target: $crate::APP_NAME, $($arg)+)
    };
}
