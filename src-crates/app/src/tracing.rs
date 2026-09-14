//! Application tracing setup.

use tracing::level_filters::LevelFilter;
use tracing_subscriber::{EnvFilter, filter::Targets, prelude::*};

const DEFAULT_LOG_LEVEL: &str = "warn";

/// Supported runtime log levels.
pub(crate) const LOG_LEVELS: [&str; 5] =
    ["trace", "debug", "info", "warn", "error"];

/// Initializes project-only tracing at the given log level.
pub(crate) fn setup_tracing(app_name: &str, log_level: Option<&str>) {
    let filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::from_default_env()
    } else {
        let level = log_level.unwrap_or(DEFAULT_LOG_LEVEL);
        EnvFilter::new(format!("off,{app_name}={level}"))
    };

    tracing_subscriber::registry()
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_filter(
                    Targets::new().with_target(app_name, LevelFilter::TRACE),
                ),
        )
        .init();
}
