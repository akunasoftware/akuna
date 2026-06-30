//! Detects reading-order layout blocks (text, title, list, table, figure) from
//! page images, independently of OCR.
//!
//! # Models
//!
//! Select a checkpoint via [`LayoutModel`][crate::layout::LayoutModel]
//! (defaults to `PpDocLayoutV3`):
//!
//! - `PpDocLayoutV3` — `PaddlePaddle/PP-DocLayoutV3_safetensors`
//!
//! # Example
//!
//! ```rust,no_run
//! use akuna_core::layout::{LayoutDetector, LayoutOptions};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let detector = LayoutDetector::new(LayoutOptions::default()).await?;
//! # Ok(())
//! # }
//! ```

use std::path::PathBuf;

use burn_dispatch::DispatchDevice;
use image::DynamicImage;

mod models;

use crate::layout::models::pp_doclayout::{
    PpDocLayoutRuntime, load_pp_doclayout_runtime,
};
use crate::ml::backend::{self, Backend};

/// Supported document layout model checkpoints.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LayoutModel {
    /// `PaddlePaddle/PP-DocLayoutV3_safetensors`.
    #[default]
    PpDocLayoutV3,
}

/// Layout detector options.
#[derive(Debug, Clone, Default)]
pub struct LayoutOptions {
    /// Which layout checkpoint to load.
    pub model: LayoutModel,
    /// Optional model download cache directory.
    pub cache_dir: Option<PathBuf>,
}

/// Layout detection failure.
#[derive(Debug, thiserror::Error)]
pub enum LayoutError {
    /// Layout model files or weights failed to load.
    #[error("Layout model load failed")]
    Load {
        /// Underlying loader error.
        source: anyhow::Error,
    },

    /// Layout preprocessing or inference failed.
    #[error("Layout detection failed")]
    Detect {
        /// Underlying detection error.
        source: anyhow::Error,
    },
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
}

impl LayoutDetector {
    /// Loads the layout detector from `options`.
    pub async fn new(options: LayoutOptions) -> Result<Self, LayoutError> {
        Self::new_on(backend::active_device(), options).await
    }

    /// Loads the layout detector from `options` onto a specific device.
    pub(crate) async fn new_on(
        device: DispatchDevice,
        options: LayoutOptions,
    ) -> Result<Self, LayoutError> {
        let runtime = match options.model {
            LayoutModel::PpDocLayoutV3 => {
                load_pp_doclayout_runtime(&device, options.cache_dir)
                    .await
                    .map_err(|source| LayoutError::Load { source })?
            }
        };

        Ok(Self { runtime, device })
    }

    /// Detects layout blocks from a decoded image.
    pub fn detect_image(
        &self,
        image: &DynamicImage,
    ) -> Result<LayoutPage, LayoutError> {
        let blocks = self
            .runtime
            .detect_image(image, &self.device)
            .map_err(|source| LayoutError::Detect { source })?
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
}
