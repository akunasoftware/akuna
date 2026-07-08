//! OCR bindings.
//!
//! ```rust,no_run
//! # async fn example() -> Result<(), akuna_ffi::ocr::OcrError> {
//! let ocr = akuna_ffi::ocr::load_ocr_engine(None).await?;
//! let _pipeline = ocr.pipeline();
//! # Ok(())
//! # }
//! ```

use std::path::{Path, PathBuf};

use akuna_core::ocr as core_ocr;

/// OCR adapter error.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum OcrError {
    /// Model load failure.
    #[error("{message}")]
    Load {
        /// Human-readable error message.
        message: String,
    },
    /// File read failure.
    #[error("{message}")]
    ReadFile {
        /// Human-readable error message.
        message: String,
    },
    /// Image decode failure.
    #[error("{message}")]
    DecodeImage {
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

/// OCR detection model checkpoint.
#[derive(uniffi::Enum)]
pub enum OcrDetectionModel {
    /// PaddleOCR PP-OCRv6 tiny detector.
    PpOcrV6Tiny,
    /// PaddleOCR PP-OCRv6 small detector.
    PpOcrV6Small,
    /// PaddleOCR PP-OCRv6 medium detector.
    PpOcrV6Medium,
}

/// OCR recognition model checkpoint.
#[derive(uniffi::Enum)]
pub enum OcrRecognitionModel {
    /// PaddleOCR PP-OCRv6 tiny recognizer.
    PpOcrV6Tiny,
    /// PaddleOCR PP-OCRv6 small recognizer.
    PpOcrV6Small,
    /// PaddleOCR PP-OCRv6 medium recognizer.
    PpOcrV6Medium,
}

/// OCR block kind.
#[derive(uniffi::Enum)]
pub enum OcrBlockKind {
    /// Text block.
    Text,
    /// Unknown block kind.
    Unknown,
}

/// OCR options.
#[derive(uniffi::Record)]
pub struct OcrEngineOptions {
    /// Region detector model.
    pub detection_model: OcrDetectionModel,
    /// Text recognizer model.
    pub recognition_model: OcrRecognitionModel,
    /// Optional model download cache directory.
    pub cache_dir: Option<String>,
}

/// Configured OCR model pipeline.
#[derive(uniffi::Record)]
pub struct OcrPipeline {
    /// Region detector model.
    pub detection_model: OcrDetectionModel,
    /// Text recognizer model.
    pub recognition_model: OcrRecognitionModel,
}

/// OCR output for one page image.
#[derive(uniffi::Record)]
pub struct OcrPage {
    /// Source image width in pixels.
    pub width: u32,
    /// Source image height in pixels.
    pub height: u32,
    /// OCR blocks in reading order.
    pub blocks: Vec<OcrBlock>,
}

/// OCR text block.
#[derive(uniffi::Record)]
pub struct OcrBlock {
    /// Recognized text.
    pub text: String,
    /// Bounding box.
    pub bbox: OcrRect,
    /// Recognition confidence.
    pub confidence: Option<f32>,
    /// OCR block kind.
    pub kind: OcrBlockKind,
}

/// OCR bounding rectangle.
#[derive(uniffi::Record)]
pub struct OcrRect {
    /// Left coordinate in pixels.
    pub x: f32,
    /// Top coordinate in pixels.
    pub y: f32,
    /// Width in pixels.
    pub width: f32,
    /// Height in pixels.
    pub height: f32,
}

/// OCR engine.
#[derive(uniffi::Object)]
pub struct OcrEngine {
    inner: core_ocr::OcrEngine,
}

#[uniffi::export(async_runtime = "tokio")]
/// Loads OCR models.
pub async fn load_ocr_engine(
    options: Option<OcrEngineOptions>,
) -> Result<OcrEngine, OcrError> {
    let options = options.map(Into::into).unwrap_or_default();
    let inner = crate::stack::run_async(core_ocr::OcrEngine::new(options))
        .map_err(inference_error)?
        .map_err(OcrError::from)?;
    Ok(OcrEngine { inner })
}

#[uniffi::export]
impl OcrEngine {
    /// Extracts OCR text from an image path.
    pub fn extract_path(&self, path: String) -> Result<OcrPage, OcrError> {
        crate::stack::run(|| self.inner.extract_file(Path::new(&path)))
            .map_err(inference_error)?
            .map(OcrPage::from)
            .map_err(OcrError::from)
    }

    /// Extracts OCR text from encoded image bytes.
    pub fn extract_bytes(&self, data: Vec<u8>) -> Result<OcrPage, OcrError> {
        crate::stack::run(|| self.inner.extract_bytes(&data))
            .map_err(inference_error)?
            .map(OcrPage::from)
            .map_err(OcrError::from)
    }

    /// Returns the configured detection and recognition models.
    pub fn pipeline(&self) -> OcrPipeline {
        self.inner.pipeline().into()
    }
}

impl From<OcrEngineOptions> for core_ocr::OcrEngineOptions {
    fn from(value: OcrEngineOptions) -> Self {
        Self {
            detection_model: value.detection_model.into(),
            recognition_model: value.recognition_model.into(),
            cache_dir: value.cache_dir.map(PathBuf::from),
        }
    }
}

impl From<OcrDetectionModel> for core_ocr::OcrDetectionModel {
    fn from(value: OcrDetectionModel) -> Self {
        match value {
            OcrDetectionModel::PpOcrV6Tiny => Self::PpOcrV6Tiny,
            OcrDetectionModel::PpOcrV6Small => Self::PpOcrV6Small,
            OcrDetectionModel::PpOcrV6Medium => Self::PpOcrV6Medium,
        }
    }
}

impl From<core_ocr::OcrDetectionModel> for OcrDetectionModel {
    fn from(value: core_ocr::OcrDetectionModel) -> Self {
        match value {
            core_ocr::OcrDetectionModel::PpOcrV6Tiny => Self::PpOcrV6Tiny,
            core_ocr::OcrDetectionModel::PpOcrV6Small => Self::PpOcrV6Small,
            core_ocr::OcrDetectionModel::PpOcrV6Medium => Self::PpOcrV6Medium,
        }
    }
}

impl From<OcrRecognitionModel> for core_ocr::OcrRecognitionModel {
    fn from(value: OcrRecognitionModel) -> Self {
        match value {
            OcrRecognitionModel::PpOcrV6Tiny => Self::PpOcrV6Tiny,
            OcrRecognitionModel::PpOcrV6Small => Self::PpOcrV6Small,
            OcrRecognitionModel::PpOcrV6Medium => Self::PpOcrV6Medium,
        }
    }
}

impl From<core_ocr::OcrRecognitionModel> for OcrRecognitionModel {
    fn from(value: core_ocr::OcrRecognitionModel) -> Self {
        match value {
            core_ocr::OcrRecognitionModel::PpOcrV6Tiny => Self::PpOcrV6Tiny,
            core_ocr::OcrRecognitionModel::PpOcrV6Small => Self::PpOcrV6Small,
            core_ocr::OcrRecognitionModel::PpOcrV6Medium => Self::PpOcrV6Medium,
        }
    }
}

impl From<core_ocr::OcrPage> for OcrPage {
    fn from(value: core_ocr::OcrPage) -> Self {
        Self {
            width: value.width,
            height: value.height,
            blocks: value.blocks.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<core_ocr::OcrBlock> for OcrBlock {
    fn from(value: core_ocr::OcrBlock) -> Self {
        Self {
            text: value.text,
            bbox: value.bbox.into(),
            confidence: value.confidence,
            kind: value.kind.into(),
        }
    }
}

impl From<core_ocr::OcrRect> for OcrRect {
    fn from(value: core_ocr::OcrRect) -> Self {
        Self {
            x: value.x,
            y: value.y,
            width: value.width,
            height: value.height,
        }
    }
}

impl From<core_ocr::OcrPipeline> for OcrPipeline {
    fn from(value: core_ocr::OcrPipeline) -> Self {
        Self {
            detection_model: value.detection_model.into(),
            recognition_model: value.recognition_model.into(),
        }
    }
}

impl From<core_ocr::OcrBlockKind> for OcrBlockKind {
    fn from(value: core_ocr::OcrBlockKind) -> Self {
        match value {
            core_ocr::OcrBlockKind::Text => Self::Text,
            core_ocr::OcrBlockKind::Unknown => Self::Unknown,
            _ => Self::Unknown,
        }
    }
}

impl From<core_ocr::OcrError> for OcrError {
    fn from(value: core_ocr::OcrError) -> Self {
        match value {
            core_ocr::OcrError::ReadFile { path, source } => Self::ReadFile {
                message: format!(
                    "failed to read OCR input file '{}': {source}",
                    path.display()
                ),
            },
            core_ocr::OcrError::DecodeImage { source } => Self::DecodeImage {
                message: source.to_string(),
            },
            core_ocr::OcrError::Load { source } => Self::Load {
                message: source.to_string(),
            },
            core_ocr::OcrError::Inference { source } => Self::Inference {
                message: source.to_string(),
            },
        }
    }
}

fn inference_error(error: impl ToString) -> OcrError {
    OcrError::Inference {
        message: error.to_string(),
    }
}
