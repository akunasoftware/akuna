//! Image OCR and document layout analysis.
//!
//! Configure the pipeline with `OcrDetectionModel` and `OcrRecognitionModel`;
//! any detection model may be paired with any recognition model. Domain
//! extraction structures live in [`crate::extraction`].
//!
//! # Example
//!
//! ```rust,no_run
//! use akuna_core::ocr::{OcrEngine, OcrEngineOptions};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let ocr = OcrEngine::new(OcrEngineOptions::default()).await?;
//! # Ok(())
//! # }
//! ```

pub mod layout;

mod error;
mod models;
mod output;

use std::path::Path;
use std::sync::Arc;

use burn_dispatch::DispatchDevice;
use heif_oxide::Pixels;
use image::{DynamicImage, ImageBuffer};
use tokio::sync::Mutex;

use crate::ml::{
    backend::{self, Backend},
    boxed_model_error,
};
use crate::ocr::models::pp_ocr::runtime::PpOcrRuntime;
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
pub enum OcrDetectionModel {
    /// `PaddlePaddle/PP-OCRv6_tiny_det_safetensors`.
    #[default]
    #[serde(alias = "PpOcrV6TinyDet")]
    PpOcrV6Tiny,
    /// `PaddlePaddle/PP-OCRv6_small_det_safetensors`.
    #[serde(alias = "PpOcrV6SmallDet")]
    PpOcrV6Small,
    /// `PaddlePaddle/PP-OCRv6_medium_det_safetensors`.
    #[serde(alias = "PpOcrV6MediumDet")]
    PpOcrV6Medium,
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
pub enum OcrRecognitionModel {
    /// `PaddlePaddle/PP-OCRv6_tiny_rec_safetensors`.
    #[default]
    #[serde(alias = "PpOcrV6TinyRec")]
    PpOcrV6Tiny,
    /// `PaddlePaddle/PP-OCRv6_small_rec_safetensors`.
    #[serde(alias = "PpOcrV6SmallRec")]
    PpOcrV6Small,
    /// `PaddlePaddle/PP-OCRv6_medium_rec_safetensors`.
    #[serde(alias = "PpOcrV6MediumRec")]
    PpOcrV6Medium,
}

impl std::fmt::Display for OcrDetectionModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::fmt::Display for OcrRecognitionModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Options for OCR model loading and inference.
#[derive(
    Debug, Clone, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize,
)]
pub struct OcrEngineOptions {
    /// Region detector used before recognition.
    #[serde(alias = "detector")]
    pub detection_model: OcrDetectionModel,
    /// Text recognizer used after detection.
    #[serde(alias = "recognizer")]
    pub recognition_model: OcrRecognitionModel,
    /// Optional model download cache directory.
    pub cache_dir: Option<std::path::PathBuf>,
}

impl OcrEngineOptions {
    fn effective(self) -> Result<Self, OcrError> {
        // Match the loader's ApiBuilder::new() default, not HF_HOME.
        let cache_dir = self
            .cache_dir
            .unwrap_or_else(|| hf_hub::Cache::default().path().clone());
        // Resolve before awaiting so a later cwd change cannot redirect loading.
        let cache_dir = std::path::absolute(cache_dir).map_err(|source| {
            OcrError::Load {
                source: Box::new(source),
            }
        })?;
        Ok(Self {
            cache_dir: Some(cache_dir),
            ..self
        })
    }
}

/// Configured OCR detection and recognition models.
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
pub struct OcrPipeline {
    /// Region detector used before recognition.
    pub detection_model: OcrDetectionModel,
    /// Text recognizer used after detection.
    pub recognition_model: OcrRecognitionModel,
}

/// OCR engine that detects and recognizes text in page images.
pub struct OcrEngine {
    model: Arc<PpOcrRuntime<Backend>>,
    device: DispatchDevice,
}

static DEFAULT_MODELS: Mutex<
    Vec<(OcrEngineOptions, Arc<PpOcrRuntime<Backend>>)>,
> = Mutex::const_new(Vec::new());

impl OcrEngine {
    /// Shares process-retained models for matching options on the default device.
    pub async fn new(options: OcrEngineOptions) -> Result<Self, OcrError> {
        let options = options.effective()?;
        // ponytail: serialize initialization across options; split locks only if
        // cold-load contention matters. Inference never takes this lock.
        let mut models = DEFAULT_MODELS.lock().await;
        if let Some((_, model)) = models.iter().find(|(key, _)| key == &options)
        {
            return Ok(Self {
                model: Arc::clone(model),
                device: backend::active_device(),
            });
        }

        // Publish only complete loads; errors and cancellation allow retry.
        let engine =
            Self::new_on(backend::active_device(), options.clone()).await?;
        models.push((options, Arc::clone(&engine.model)));
        Ok(engine)
    }

    /// Loads OCR models from `options` onto a specific device.
    pub(crate) async fn new_on(
        device: DispatchDevice,
        options: OcrEngineOptions,
    ) -> Result<Self, OcrError> {
        let model = PpOcrRuntime::load(
            options.detection_model,
            options.recognition_model,
            &device,
            options.cache_dir,
        )
        .await
        .map_err(|source| OcrError::Load {
            source: boxed_model_error(source),
        })?;

        Ok(Self {
            model: Arc::new(model),
            device,
        })
    }

    /// Extracts OCR blocks from an image file.
    pub fn extract_file(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<OcrPage, OcrError> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).map_err(|source| OcrError::ReadFile {
                path: path.to_path_buf(),
                source,
            })?;

        self.extract_bytes(&bytes)
    }

    /// Extracts OCR blocks from encoded image bytes.
    pub fn extract_bytes(&self, bytes: &[u8]) -> Result<OcrPage, OcrError> {
        let image = if crate::heif::has_heif_brand(bytes) {
            heif_image(bytes)?
        } else {
            image::load_from_memory(bytes)
                .map_err(|source| OcrError::DecodeImage { source })?
        };
        self.model
            .extract_page(&image, &self.device)
            .map_err(|source| OcrError::Inference {
                source: boxed_model_error(source),
            })
    }

    /// Returns the configured detection and recognition models.
    pub fn pipeline(&self) -> OcrPipeline {
        OcrPipeline {
            detection_model: self.model.detection_model,
            recognition_model: self.model.recognition_model,
        }
    }
}

/// Decodes HEIC/HEIF bytes into an image for OCR.
///
/// The `image` crate has no HEIC support, so brand-tagged bytes are decoded
/// through the pure-Rust HEIF decoder instead.
fn heif_image(bytes: &[u8]) -> Result<DynamicImage, OcrError> {
    let decoded = heif_oxide::decode_bytes(bytes)
        .map_err(|source| OcrError::HeifDecodeImage { source })?;
    let image = match decoded.pixels {
        Pixels::Rgb8(values) => DynamicImage::ImageRgb8(pixel_buffer(
            decoded.width,
            decoded.height,
            values,
        )?),
        Pixels::Rgba8(values) => DynamicImage::ImageRgba8(pixel_buffer(
            decoded.width,
            decoded.height,
            values,
        )?),
        Pixels::Rgb16(values) => DynamicImage::ImageRgb16(pixel_buffer(
            decoded.width,
            decoded.height,
            values,
        )?),
        Pixels::Rgba16(values) => DynamicImage::ImageRgba16(pixel_buffer(
            decoded.width,
            decoded.height,
            values,
        )?),
    };
    Ok(image)
}

/// Wraps a decoded pixel vector, rejecting length mismatches.
fn pixel_buffer<P, C>(
    width: u32,
    height: u32,
    values: Vec<P>,
) -> Result<ImageBuffer<C, Vec<P>>, OcrError>
where
    C: image::Pixel<Subpixel = P>,
{
    ImageBuffer::from_raw(width, height, values).ok_or(OcrError::InvalidImage)
}

#[cfg(test)]
mod tests;
