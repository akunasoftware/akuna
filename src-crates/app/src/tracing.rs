//! Application tracing setup.

use tracing_subscriber::EnvFilter;

const DEFAULT_LOG_LEVEL: &str = "warn";

/// Supported runtime log levels.
pub(crate) const LOG_LEVELS: [&str; 5] =
    ["trace", "debug", "info", "warn", "error"];

/// Initializes application tracing for `app_name` at the given log level.
pub(crate) fn setup_tracing(app_name: &str, log_level: Option<&str>) {
    let filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::from_default_env()
    } else {
        let level = log_level.unwrap_or(DEFAULT_LOG_LEVEL);
        EnvFilter::new(format!("{app_name}={level}"))
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}
