//! Image OCR engines and OCR-specific result geometry built with Burn.
//!
//! Runs a detector + recognizer pipeline over page images. Domain extraction
//! structures live in [`crate::extraction`].
//!
//! # Models
//!
//! Region detection via [`OcrDetector`][crate::ocr::OcrDetector]
//! (defaults to `PpOcrV6MediumDet`):
//!
//! - `PpOcrV6TinyDet` / `PpOcrV6SmallDet` / `PpOcrV6MediumDet` — PaddleOCR PP-OCRv6 detectors
//!
//! Text recognition via [`OcrRecognizer`][crate::ocr::OcrRecognizer]
//! (defaults to `PpOcrV6MediumRec`):
//!
//! - `PpOcrV6TinyRec` / `PpOcrV6SmallRec` / `PpOcrV6MediumRec` — PaddleOCR PP-OCRv6 recognizers
//!
//! Any detector may be paired with any recognizer.
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

use burn::tensor::backend::Backend;
use burn_wgpu::{Wgpu, WgpuDevice};

use self::models::pp_ocr::runtime::PpOcrRuntime;
pub use error::OcrError;
pub use output::{OcrBlock, OcrBlockKind, OcrPage, OcrRect};

/// Default OCR backend.
pub(crate) type DefaultBackend = Wgpu;

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

/// Minimal OCR interface for file and byte extraction.
#[derive(Debug)]
pub struct Ocr<B: Backend = DefaultBackend> {
    model: Box<PpOcrRuntime<B>>,
    device: B::Device,
}

impl Ocr<DefaultBackend> {
    /// Loads OCR model onto default WGPU device.
    pub async fn new(options: OcrOptions) -> Result<Self, OcrError> {
        let device = WgpuDevice::default();
        Self::new_with_device(&device, options).await
    }
}

impl<B> Ocr<B>
where
    B: Backend<FloatElem = f32>,
{
    /// Loads OCR model onto provided device.
    pub async fn new_with_device(
        device: &B::Device,
        options: OcrOptions,
    ) -> Result<Self, OcrError> {
        let model = PpOcrRuntime::load(
            options.detector,
            options.recognizer,
            device,
            options.cache_dir,
        )
        .await
        .map_err(|source| OcrError::Load { source })?;

        Ok(Self {
            model: Box::new(model),
            device: device.clone(),
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

    /// Returns configured detector and recognizer.
    pub fn pipeline(&self) -> (OcrDetector, OcrRecognizer) {
        (self.model.detector, self.model.recognizer)
    }
}
