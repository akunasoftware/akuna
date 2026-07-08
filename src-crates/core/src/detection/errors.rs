type DetectionErrorSource = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Errors raised while detecting file types.
#[derive(Debug, thiserror::Error)]
pub enum DetectionError {
    /// Reading an input file failed.
    #[error("I/O error: {source}")]
    Io {
        /// The underlying I/O error.
        #[from]
        source: std::io::Error,
    },

    /// Loading or running the bundled model failed.
    #[error("model {operation} failed: {source}")]
    Model {
        /// The operation that failed.
        operation: &'static str,
        /// The underlying model error.
        #[source]
        source: DetectionErrorSource,
    },

    /// The bundled model produced invalid data.
    #[error("invalid model data: {message}")]
    InvalidModel {
        /// Details of the invalid data.
        message: String,
    },
}
