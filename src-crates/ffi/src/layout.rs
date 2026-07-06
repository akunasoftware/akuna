//! Layout bindings.

use std::path::Path;

use akuna_core::layout as core_layout;

/// Layout adapter error.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum LayoutError {
    /// Model load failure.
    #[error("{message}")]
    Load {
        /// Human-readable error message.
        message: String,
    },
    /// Image read failure.
    #[error("{message}")]
    ReadImage {
        /// Human-readable error message.
        message: String,
    },
    /// Image decode failure.
    #[error("{message}")]
    DecodeImage {
        /// Human-readable error message.
        message: String,
    },
    /// Layout detection failure.
    #[error("{message}")]
    Detect {
        /// Human-readable error message.
        message: String,
    },
}

/// Supported layout model checkpoints.
#[derive(uniffi::Enum)]
pub enum LayoutModel {
    /// `PaddlePaddle/PP-DocLayoutV3_safetensors`.
    PpDocLayoutV3,
}

/// Layout detector options.
#[derive(uniffi::Record)]
pub struct LayoutDetectorOptions {
    /// Model checkpoint to load.
    pub model: LayoutModel,
}

/// Layout output for one page image.
#[derive(uniffi::Record)]
pub struct LayoutPage {
    /// Source image width in pixels.
    pub width: u32,
    /// Source image height in pixels.
    pub height: u32,
    /// Detected layout blocks.
    pub blocks: Vec<LayoutBlock>,
}

/// Detected layout block.
#[derive(uniffi::Record)]
pub struct LayoutBlock {
    /// Detector label.
    pub label: String,
    /// Detector confidence.
    pub confidence: f32,
    /// Bounding box.
    pub bbox: LayoutRect,
    /// Reading order.
    pub order: i64,
}

/// Layout bounding rectangle.
#[derive(uniffi::Record)]
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

/// Document layout detector.
#[derive(uniffi::Object)]
pub struct LayoutDetector {
    inner: core_layout::LayoutDetector,
}

#[uniffi::export(async_runtime = "tokio")]
/// Loads a layout detector.
pub async fn load_layout_detector(
    options: Option<LayoutDetectorOptions>,
) -> Result<LayoutDetector, LayoutError> {
    let options = options.map(Into::into).unwrap_or_default();
    let inner = core_layout::LayoutDetector::new(options)
        .await
        .map_err(LayoutError::from)?;
    Ok(LayoutDetector { inner })
}

#[uniffi::export]
impl LayoutDetector {
    /// Detects layout blocks from an image path.
    pub fn detect_path(&self, path: String) -> Result<LayoutPage, LayoutError> {
        let image = image::open(Path::new(&path)).map_err(image_error)?;
        crate::stack::run(|| self.inner.detect_image(&image))
            .map_err(detect_error)?
            .map(LayoutPage::from)
            .map_err(LayoutError::from)
    }
}

impl From<LayoutDetectorOptions> for core_layout::LayoutDetectorOptions {
    fn from(value: LayoutDetectorOptions) -> Self {
        Self {
            model: value.model.into(),
            cache_dir: None,
        }
    }
}

impl From<LayoutModel> for core_layout::LayoutModel {
    fn from(value: LayoutModel) -> Self {
        match value {
            LayoutModel::PpDocLayoutV3 => Self::PpDocLayoutV3,
        }
    }
}

impl From<core_layout::LayoutPage> for LayoutPage {
    fn from(value: core_layout::LayoutPage) -> Self {
        Self {
            width: value.width,
            height: value.height,
            blocks: value.blocks.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<core_layout::LayoutBlock> for LayoutBlock {
    fn from(value: core_layout::LayoutBlock) -> Self {
        Self {
            label: value.label,
            confidence: value.confidence,
            bbox: value.bbox.into(),
            order: value.order,
        }
    }
}

impl From<core_layout::LayoutRect> for LayoutRect {
    fn from(value: core_layout::LayoutRect) -> Self {
        Self {
            x: value.x,
            y: value.y,
            width: value.width,
            height: value.height,
        }
    }
}

impl From<core_layout::LayoutError> for LayoutError {
    fn from(value: core_layout::LayoutError) -> Self {
        match value {
            core_layout::LayoutError::Load { source } => Self::Load {
                message: source.to_string(),
            },
            core_layout::LayoutError::Detect { source } => Self::Detect {
                message: source.to_string(),
            },
        }
    }
}

fn image_error(error: image::ImageError) -> LayoutError {
    match error {
        image::ImageError::IoError(error) => LayoutError::ReadImage {
            message: error.to_string(),
        },
        error => LayoutError::DecodeImage {
            message: error.to_string(),
        },
    }
}

fn detect_error(error: impl ToString) -> LayoutError {
    LayoutError::Detect {
        message: error.to_string(),
    }
}
