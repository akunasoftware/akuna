//! Detects reading-order layout blocks (text, title, list, table, figure) from
//! page images.
//!
//! # Example
//!
//! ```rust,no_run
//! use akuna_core::ocr::layout::{LayoutDetector, LayoutDetectorOptions};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let detector = LayoutDetector::new(LayoutDetectorOptions::default()).await?;
//! # Ok(())
//! # }
//! ```

use std::path::{Path, PathBuf};

use burn_dispatch::DispatchDevice;

mod error;
mod models;

use crate::ml::{
    backend::{self, Backend},
    boxed_model_error,
};
use crate::ocr::layout::models::pp_doclayout::{
    PpDocLayoutRuntime, load_pp_doclayout_runtime,
};

pub use error::LayoutError;

/// Supported document layout model checkpoints.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LayoutModel {
    /// `PaddlePaddle/PP-DocLayoutV3_safetensors` at `97d101e`.
    #[default]
    PpDocLayoutV3,
}

/// Layout detector options.
#[derive(Debug, Clone, Default)]
pub struct LayoutDetectorOptions {
    /// Which layout checkpoint to load.
    pub model: LayoutModel,
    /// Optional model download cache directory.
    pub cache_dir: Option<PathBuf>,
}

/// Layout output for one page/image.
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Deserialize,
    serde::Serialize,
    utoipa::ToSchema,
)]
pub struct LayoutPage {
    /// Source image width in pixels.
    pub width: u32,
    /// Source image height in pixels.
    pub height: u32,
    /// Detected layout blocks in reading order.
    pub blocks: Vec<LayoutBlock>,
}

/// Detected layout block.
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Deserialize,
    serde::Serialize,
    utoipa::ToSchema,
)]
pub struct LayoutBlock {
    /// Detector label.
    pub label: String,
    /// Detector confidence from 0 to 1.
    pub confidence: f32,
    /// Bounding box in image pixel coordinates.
    pub bbox: LayoutRect,
    /// Detector reading order.
    pub order: i64,
}

/// Axis-aligned layout bounding box.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    serde::Deserialize,
    serde::Serialize,
    utoipa::ToSchema,
)]
pub struct LayoutRect {
    /// Left coordinate in pixels.
    pub x: f32,
    /// Top coordinate in pixels.
    pub y: f32,
    /// Width in pixels.
    pub width: f32,
    /// Height in pixels.
    pub height: f32,
}

/// Detects document layout blocks from page images.
pub struct LayoutDetector {
    runtime: PpDocLayoutRuntime<Backend>,
    device: DispatchDevice,
    model: LayoutModel,
}

impl LayoutDetector {
    /// Loads the layout detector from `options`.
    pub async fn new(
        options: LayoutDetectorOptions,
    ) -> Result<Self, LayoutError> {
        Self::new_on(backend::active_device(), options).await
    }

    /// Loads the layout detector from `options` onto a specific device.
    pub(crate) async fn new_on(
        device: DispatchDevice,
        options: LayoutDetectorOptions,
    ) -> Result<Self, LayoutError> {
        let LayoutDetectorOptions { model, cache_dir } = options;
        let runtime = match model {
            LayoutModel::PpDocLayoutV3 => {
                load_pp_doclayout_runtime(&device, cache_dir)
                    .await
                    .map_err(|source| LayoutError::Load {
                        source: boxed_model_error(source),
                    })?
            }
        };

        Ok(Self {
            runtime,
            device,
            model,
        })
    }

    /// Detects layout blocks from an image file.
    pub fn detect_file(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<LayoutPage, LayoutError> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).map_err(|source| LayoutError::ReadFile {
                path: path.to_path_buf(),
                source,
            })?;

        self.detect_bytes(&bytes)
    }

    /// Detects layout blocks from encoded image bytes.
    pub fn detect_bytes(
        &self,
        bytes: &[u8],
    ) -> Result<LayoutPage, LayoutError> {
        let image = image::load_from_memory(bytes)
            .map_err(|source| LayoutError::DecodeImage { source })?;

        self.detect_decoded(&image)
    }

    fn detect_decoded(
        &self,
        image: &image::DynamicImage,
    ) -> Result<LayoutPage, LayoutError> {
        let blocks = self
            .runtime
            .detect_image(image, &self.device)
            .map_err(|source| LayoutError::Detect {
                source: boxed_model_error(source),
            })?
            .into_iter()
            .map(|detection| LayoutBlock {
                label: detection.label,
                confidence: detection.score,
                bbox: LayoutRect {
                    x: detection.bbox[0],
                    y: detection.bbox[1],
                    width: detection.bbox[2] - detection.bbox[0],
                    height: detection.bbox[3] - detection.bbox[1],
                },
                order: detection.order,
            })
            .collect();

        Ok(LayoutPage {
            width: image.width(),
            height: image.height(),
            blocks,
        })
    }

    /// Returns the loaded layout checkpoint.
    pub fn model(&self) -> LayoutModel {
        self.model
    }
}

#[cfg(test)]
mod tests;
