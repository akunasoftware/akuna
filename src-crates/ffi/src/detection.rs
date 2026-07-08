//! Detection bindings.
//!
//! ```rust,no_run
//! let detector = akuna_ffi::detection::FileTypeDetector::new()?;
//! let _file_type = detector.identify_bytes(b"plain text".to_vec())?;
//! # Ok::<(), akuna_ffi::detection::DetectionError>(())
//! ```

use std::path::Path;

use akuna_core::detection as core_detection;

/// Detection failure.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum DetectionError {
    /// Detection failure.
    #[error("{message}")]
    Runtime {
        /// Human-readable error message.
        message: String,
    },
}

/// The source that resolved a file type.
#[derive(uniffi::Enum)]
pub enum DetectionOrigin {
    /// A deterministic rule resolved the type.
    Rule,
    /// The bundled model resolved the type.
    Model,
}

/// Metadata describing a detected file type.
#[derive(uniffi::Record)]
pub struct FileTypeInfo {
    /// Unique file type label.
    pub label: String,
    /// MIME type.
    pub mime_type: String,
    /// Type group.
    pub group: String,
    /// Human-readable description.
    pub description: String,
    /// Known filename extensions.
    pub extensions: Vec<String>,
    /// Whether the file type is text-like.
    pub is_text: bool,
}

/// Identified file type.
#[derive(uniffi::Record)]
pub struct FileType {
    /// File type metadata.
    pub info: FileTypeInfo,
    /// Detection confidence score.
    pub confidence: f32,
    /// Whether a rule or model resolved this type.
    pub origin: DetectionOrigin,
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
        let inner = crate::stack::run(core_detection::FileTypeDetector::new)
            .map_err(to_error)?
            .map_err(to_error)?;
        Ok(Self { inner })
    }

    /// Identifies the file type of raw bytes.
    pub fn identify_bytes(
        &self,
        data: Vec<u8>,
    ) -> Result<FileType, DetectionError> {
        crate::stack::run(|| self.inner.identify_bytes(&data))
            .map_err(to_error)?
            .map(FileType::from)
            .map_err(to_error)
    }

    /// Identifies the file type of a file path.
    pub fn identify_path(
        &self,
        path: String,
    ) -> Result<FileType, DetectionError> {
        crate::stack::run(|| self.inner.identify_file(Path::new(&path)))
            .map_err(to_error)?
            .map(FileType::from)
            .map_err(to_error)
    }
}

impl From<core_detection::FileType> for FileType {
    fn from(value: core_detection::FileType) -> Self {
        let info = value.info();
        Self {
            info: FileTypeInfo {
                label: info.label.clone(),
                mime_type: info.mime_type.clone(),
                group: info.group.clone(),
                description: info.description.clone(),
                extensions: info.extensions.clone(),
                is_text: info.is_text,
            },
            confidence: value.confidence(),
            origin: value.origin().into(),
        }
    }
}

impl From<core_detection::DetectionOrigin> for DetectionOrigin {
    fn from(value: core_detection::DetectionOrigin) -> Self {
        match value {
            core_detection::DetectionOrigin::Rule => Self::Rule,
            core_detection::DetectionOrigin::Model => Self::Model,
        }
    }
}

fn to_error(error: impl ToString) -> DetectionError {
    DetectionError::Runtime {
        message: error.to_string(),
    }
}
