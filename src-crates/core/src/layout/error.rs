use std::path::PathBuf;

/// Layout detection failure.
#[derive(Debug, thiserror::Error)]
pub enum LayoutError {
    /// Layout input file could not be read.
    #[error("failed to read layout input file '{path}'")]
    ReadFile {
        /// Path that failed to read.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// Layout input bytes were not a supported image.
    #[error("failed to decode layout input image")]
    DecodeImage {
        /// Underlying image decoder error.
        source: image::ImageError,
    },

    /// Layout model weights failed to load.
    #[error("layout model load failed")]
    Load {
        /// Underlying model loader error.
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Layout preprocessing or inference failed.
    #[error("layout detection failed")]
    Detect {
        /// Underlying detection error.
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}
