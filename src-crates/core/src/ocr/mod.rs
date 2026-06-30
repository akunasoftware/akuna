//! Image OCR: run text detection and recognition over page images.
//!
//! Configure the pipeline with `OcrDetector` and `OcrRecognizer`; any
//! detector may be paired with any recognizer. Domain extraction structures
//! live in [`crate::extraction`].
//!
//! # Example
//!
//! ```rust,no_run
//! use akuna_core::ocr::{Ocr, OcrOptions};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let ocr = Ocr::new(OcrOptions::default()).await?;
//! # Ok(())
//! # }
//! ```

mod error;
mod models;
mod output;

use std::path::Path;

use burn_dispatch::DispatchDevice;

use self::models::pp_ocr::runtime::PpOcrRuntime;
use crate::ml::backend::{self, Backend};
pub use error::OcrError;
pub use output::{OcrBlock, OcrBlockKind, OcrPage, OcrRect};

/// Region detection strategy.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    serde::Deserialize,
    serde::Serialize,
    utoipa::ToSchema,
)]
pub enum OcrDetector {
    /// PaddleOCR PP-OCRv6 tiny detector.
    PpOcrV6TinyDet,
    /// PaddleOCR PP-OCRv6 small detector.
    PpOcrV6SmallDet,
    /// PaddleOCR PP-OCRv6 medium detector.
    #[default]
    PpOcrV6MediumDet,
}

/// Text recognition strategy.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    serde::Deserialize,
    serde::Serialize,
    utoipa::ToSchema,
)]
pub enum OcrRecognizer {
    /// PaddleOCR PP-OCRv6 tiny recognizer.
    PpOcrV6TinyRec,
    /// PaddleOCR PP-OCRv6 small recognizer.
    PpOcrV6SmallRec,
    /// PaddleOCR PP-OCRv6 medium recognizer.
    #[default]
    PpOcrV6MediumRec,
}

impl std::fmt::Display for OcrDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::fmt::Display for OcrRecognizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Options for OCR model loading and inference.
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
pub struct OcrOptions {
    /// Region detector used before recognition.
    pub detector: OcrDetector,
    /// Text recognizer used after detection.
    pub recognizer: OcrRecognizer,
    /// Optional model download cache directory.
    pub cache_dir: Option<std::path::PathBuf>,
}

/// OCR engine that detects and recognizes text in page images.
pub struct Ocr {
    model: Box<PpOcrRuntime<Backend>>,
    device: DispatchDevice,
}

impl Ocr {
    /// Loads OCR models from `options` onto the auto-selected device.
    pub async fn new(options: OcrOptions) -> Result<Self, OcrError> {
        Self::new_on(backend::active_device(), options).await
    }

    /// Loads OCR models from `options` onto a specific device.
    pub(crate) async fn new_on(
        device: DispatchDevice,
        options: OcrOptions,
    ) -> Result<Self, OcrError> {
        let model = PpOcrRuntime::load(
            options.detector,
            options.recognizer,
            &device,
            options.cache_dir,
        )
        .await
        .map_err(|source| OcrError::Load { source })?;

        Ok(Self {
            model: Box::new(model),
            device,
        })
    }

    /// Extracts OCR blocks from an image file.
    pub fn extract_page_file(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<OcrPage, OcrError> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).map_err(|source| OcrError::ReadFile {
                path: path.to_path_buf(),
                source,
            })?;

        self.extract_page_bytes(&bytes)
    }

    /// Extracts OCR blocks from encoded image bytes.
    pub fn extract_page_bytes(
        &self,
        bytes: &[u8],
    ) -> Result<OcrPage, OcrError> {
        let image = image::load_from_memory(bytes)
            .map_err(|source| OcrError::DecodeImage { source })?;
        self.model
            .extract_page(&image, &self.device)
            .map_err(|source| OcrError::Inference { source })
    }

    /// Returns the configured detector and recognizer.
    pub fn pipeline(&self) -> (OcrDetector, OcrRecognizer) {
        (self.model.detector, self.model.recognizer)
    }
}
