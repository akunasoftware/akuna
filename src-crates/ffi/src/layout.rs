//! Layout bindings.
//!
//! ```rust,no_run
//! # async fn example() -> Result<(), akuna_ffi::layout::LayoutError> {
//! let detector = akuna_ffi::layout::load_layout_detector(None).await?;
//! let _page = detector.detect_bytes(Vec::new())?;
//! # Ok(())
//! # }
//! ```

use std::path::{Path, PathBuf};

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
    /// Optional model download cache directory.
    pub cache_dir: Option<String>,
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
    let inner =
        crate::stack::run_async(core_layout::LayoutDetector::new(options))
            .map_err(detect_error)?
            .map_err(LayoutError::from)?;
    Ok(LayoutDetector { inner })
}

#[uniffi::export]
impl LayoutDetector {
    /// Detects layout blocks from an image path.
    pub fn detect_path(&self, path: String) -> Result<LayoutPage, LayoutError> {
        crate::stack::run(|| self.inner.detect_file(Path::new(&path)))
            .map_err(detect_error)?
            .map(LayoutPage::from)
            .map_err(LayoutError::from)
    }

    /// Detects layout blocks from encoded image bytes.
    pub fn detect_bytes(
        &self,
        data: Vec<u8>,
    ) -> Result<LayoutPage, LayoutError> {
        crate::stack::run(|| self.inner.detect_bytes(&data))
            .map_err(detect_error)?
            .map(LayoutPage::from)
            .map_err(LayoutError::from)
    }

    /// Returns the loaded layout checkpoint.
    pub fn model(&self) -> LayoutModel {
        self.inner.model().into()
    }
}

impl From<LayoutDetectorOptions> for core_layout::LayoutDetectorOptions {
    fn from(value: LayoutDetectorOptions) -> Self {
        Self {
            model: value.model.into(),
            cache_dir: value.cache_dir.map(PathBuf::from),
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

impl From<core_layout::LayoutModel> for LayoutModel {
    fn from(value: core_layout::LayoutModel) -> Self {
        match value {
            core_layout::LayoutModel::PpDocLayoutV3 => Self::PpDocLayoutV3,
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
            core_layout::LayoutError::ReadFile { path, source } => {
                Self::ReadFile {
                    message: format!(
                        "failed to read layout input file '{}': {source}",
                        path.display()
                    ),
                }
            }
            core_layout::LayoutError::DecodeImage { source } => {
                Self::DecodeImage {
                    message: source.to_string(),
                }
            }
            core_layout::LayoutError::Load { source } => Self::Load {
                message: source.to_string(),
            },
            core_layout::LayoutError::Detect { source } => Self::Detect {
                message: source.to_string(),
            },
        }
    }
}

fn detect_error(error: impl ToString) -> LayoutError {
    LayoutError::Detect {
        message: error.to_string(),
    }
}
