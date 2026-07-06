//! Detection bindings.

use std::path::Path;

use akuna_core::detection as core_detection;

/// Detection adapter error.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum DetectionError {
    /// Model load failure.
    #[error("{message}")]
    Load {
        /// Human-readable error message.
        message: String,
    },
    /// File read failure.
    #[error("{message}")]
    Io {
        /// Human-readable error message.
        message: String,
    },
    /// Inference failure.
    #[error("{message}")]
    Inference {
        /// Human-readable error message.
        message: String,
    },
}

/// Identified file type metadata.
#[derive(uniffi::Record)]
pub struct FileType {
    /// Unique file type label.
    pub label: String,
    /// MIME type.
    pub mime_type: String,
    /// Type group.
    pub group: String,
    /// Human-readable description.
    pub description: String,
    /// Detection confidence score.
    pub score: f32,
}

/// File-type detector.
#[derive(uniffi::Object)]
pub struct FileTypeDetector {
    inner: core_detection::FileTypeDetector,
}

#[uniffi::export]
impl FileTypeDetector {
    /// Builds a file-type detector.
    #[uniffi::constructor]
    pub fn new() -> Result<Self, DetectionError> {
        let inner =
            core_detection::FileTypeDetector::new().map_err(load_error)?;
        Ok(Self { inner })
    }

    /// Identifies the file type of raw bytes.
    pub fn identify_bytes(
        &self,
        data: Vec<u8>,
    ) -> Result<FileType, DetectionError> {
        self.inner
            .identify_bytes(&data)
            .map(FileType::from)
            .map_err(inference_error)
    }

    /// Identifies the file type of a file path.
    pub fn identify_path(
        &self,
        path: String,
    ) -> Result<FileType, DetectionError> {
        self.inner
            .identify_file(Path::new(&path))
            .map(FileType::from)
            .map_err(DetectionError::from)
    }
}

impl From<core_detection::FileType> for FileType {
    fn from(value: core_detection::FileType) -> Self {
        let info = value.info();
        Self {
            label: info.label.to_owned(),
            mime_type: info.mime_type.to_owned(),
            group: info.group.to_owned(),
            description: info.description.to_owned(),
            score: value.score(),
        }
    }
}

impl From<core_detection::MagikaInferenceError> for DetectionError {
    fn from(value: core_detection::MagikaInferenceError) -> Self {
        match value {
            core_detection::MagikaInferenceError::Io(error) => Self::Io {
                message: error.to_string(),
            },
            core_detection::MagikaInferenceError::InvalidConfig(error) => {
                Self::Load { message: error }
            }
            core_detection::MagikaInferenceError::Runtime(error) => {
                Self::Inference { message: error }
            }
        }
    }
}

fn load_error(error: core_detection::MagikaInferenceError) -> DetectionError {
    match DetectionError::from(error) {
        DetectionError::Inference { message } => {
            DetectionError::Load { message }
        }
        error => error,
    }
}

fn inference_error(
    error: core_detection::MagikaInferenceError,
) -> DetectionError {
    match DetectionError::from(error) {
        DetectionError::Load { message } => {
            DetectionError::Inference { message }
        }
        error => error,
    }
}
