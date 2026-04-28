use std::fmt::Debug;

type ExtractionErrorSource = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Errors that can occur during file extraction.
#[derive(Debug, thiserror::Error)]
pub enum FileExtractionError {
    /// I/O error.
    #[error("I/O error: {:?}", source.to_string())]
    Io {
        /// The underlying I/O error.
        source: std::io::Error,
    },

    /// Error from an engine determining file type.
    #[error("Detection engine '{engine}' failed")]
    DetectionEngine {
        /// The name of the engine that failed.
        engine: &'static str,
        /// The underlying detection engine error.
        source: ExtractionErrorSource,
    },

    /// Error from an engine extracting file contents.
    #[error("Extraction engine '{engine}' failed")]
    ExtractionEngine {
        /// The name of the engine that failed.
        engine: &'static str,
        /// The underlying extraction engine error.
        source: ExtractionErrorSource,
    },

    /// Extraction succeeded but returned no text content.
    #[error("Extraction engine '{engine}' returned no text content")]
    MissingTextContent {
        /// The name of the engine that returned non-text content.
        engine: &'static str,
    },

    /// File at the given path has no contents.
    #[error("File has no contents")]
    NoContents,

    /// File at the given path is not supported for extraction.
    #[error(
        r#"Unsupported file:
            Mime {mime_type}
            Label {label}
            Description {description}
        "#
    )]
    UnsupportedFileType {
        /// The detected MIME type of the file.
        mime_type: String,
        /// Label of the detected MIME type (typically the file extension).
        label: String,
        /// Description of the detected MIME type.
        description: String,
    },
}

impl From<std::io::Error> for FileExtractionError {
    fn from(source: std::io::Error) -> Self {
        Self::Io { source }
    }
}
