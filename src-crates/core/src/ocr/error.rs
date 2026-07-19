use std::path::PathBuf;

/// OCR failure.
#[derive(Debug, thiserror::Error)]
pub enum OcrError {
    /// OCR input file could not be read.
    #[error("Failed to read OCR input file '{path}'")]
    ReadFile {
        /// Path that failed to read.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// OCR input bytes were not a supported image.
    #[error("Failed to decode OCR input image")]
    DecodeImage {
        /// Underlying image decoder error.
        source: image::ImageError,
    },

    /// OCR model or detector failed to load.
    #[error("OCR model load failed")]
    Load {
        /// Underlying loader error.
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// OCR preprocessing or inference failed.
    #[error("OCR inference failed")]
    Inference {
        /// Underlying OCR error.
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}
