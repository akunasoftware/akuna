//! Extraction bindings.
//!
//! ```rust,no_run
//! # async fn example() -> Result<(), akuna_ffi::extraction::ExtractionError> {
//! let _result = akuna_ffi::extraction::extract_bytes(b"text".to_vec(), None).await?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use akuna_core::extraction as core_extraction;

use crate::detection::DetectionOrigin;
use crate::ocr::OcrEngineOptions;

/// Extraction adapter error.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum ExtractionError {
    /// I/O failure.
    #[error("{message}")]
    Io {
        /// Human-readable error message.
        message: String,
    },
    /// Runtime failure.
    #[error("{message}")]
    Runtime {
        /// Human-readable error message.
        message: String,
    },
}

/// Extraction options.
#[derive(uniffi::Record)]
pub struct ExtractionOptions {
    /// Return detected metadata.
    pub return_metadata: bool,
    /// Return extracted text.
    pub return_content: bool,
    /// Return structured parts.
    pub return_parts: bool,
    /// OCR configuration for image extraction.
    pub ocr: OcrEngineOptions,
}

/// Extraction result.
#[derive(uniffi::Record)]
pub struct ExtractionResult {
    /// Detected metadata, when requested.
    pub metadata: Option<ExtractionMetadata>,
    /// Processing pipeline steps.
    pub pipeline: Vec<ExtractionPipelineStep>,
    /// Extracted text, when requested.
    pub text: Option<String>,
    /// Structured extraction parts, when requested.
    pub parts: Option<Vec<ExtractionPart>>,
}

/// Detected file metadata.
#[derive(uniffi::Record)]
pub struct ExtractionMetadata {
    /// File stem.
    pub stem: Option<String>,
    /// File extension.
    pub extension: Option<String>,
    /// Detected type label.
    pub label: String,
    /// Detected MIME type.
    pub mime_type: String,
    /// Detected type description.
    pub description: String,
    /// Whether content is text-like.
    pub is_text: bool,
    /// Detection confidence from 0 to 1.
    pub confidence: f32,
    /// Whether a rule or model resolved the file type.
    pub origin: DetectionOrigin,
    /// Content hash.
    pub hash: String,
}

/// Structured extracted part.
#[derive(uniffi::Record)]
pub struct ExtractionPart {
    /// Zero-based part index.
    pub index: u64,
    /// Semantic part kind.
    pub kind: PartKind,
    /// Extracted text.
    pub text: Option<String>,
    /// Source provenance.
    pub provenance: Option<ExtractionProvenance>,
}

/// Extracted part provenance.
#[derive(uniffi::Record)]
pub struct ExtractionProvenance {
    /// Recognition confidence.
    pub confidence: Option<f32>,
    /// One-based page number.
    pub page: Option<u64>,
    /// Source bounding box.
    pub bbox: Option<ExtractionBbox>,
}

/// Extraction bounding box.
#[derive(uniffi::Record)]
pub struct ExtractionBbox {
    /// Left coordinate.
    pub x: f32,
    /// Top coordinate.
    pub y: f32,
    /// Width.
    pub width: f32,
    /// Height.
    pub height: f32,
}

/// Extraction pipeline step.
#[derive(uniffi::Record)]
pub struct ExtractionPipelineStep {
    /// Step role.
    pub step: ExtractionPipelineStepKind,
    /// Engine name.
    pub engine: String,
    /// Step duration in milliseconds.
    pub duration_ms: u64,
    /// Step output counts.
    pub outputs: HashMap<String, u64>,
}

/// Extraction pipeline step role.
#[derive(uniffi::Enum)]
pub enum ExtractionPipelineStepKind {
    /// File type detection.
    Detection,
    /// Structured or plain content parsing.
    Parsing,
    /// Text recognition from image regions.
    Recognition,
}

/// Structured extraction part kind.
#[derive(uniffi::Enum)]
pub enum PartKind {
    /// Caption or attribution.
    Caption,
    /// Footer content.
    Footer,
    /// Heading content.
    Heading,
    /// List item.
    ListItem,
    /// Paragraph text.
    Paragraph,
    /// Table content.
    Table,
    /// Plain text.
    Text,
    /// Unknown content.
    Unknown,
}

#[uniffi::export(async_runtime = "tokio")]
/// Extracts content from a filesystem path.
pub async fn extract_path(
    path: String,
    options: Option<ExtractionOptions>,
) -> Result<ExtractionResult, ExtractionError> {
    let config = options.map(Into::into).unwrap_or_default();
    let path = PathBuf::from(path);
    run_extraction(move |handle| {
        handle.block_on(core_extraction::extract_file(&path, &config))
    })
    .await
}

#[uniffi::export(async_runtime = "tokio")]
/// Extracts content from encoded document bytes.
pub async fn extract_bytes(
    data: Vec<u8>,
    options: Option<ExtractionOptions>,
) -> Result<ExtractionResult, ExtractionError> {
    let config = options.map(Into::into).unwrap_or_default();
    run_extraction(move |handle| {
        handle.block_on(core_extraction::extract_bytes(&data, &config))
    })
    .await
}

/// Runs one core extraction on the FFI stack and converts the result.
async fn run_extraction<F>(
    extract: F,
) -> Result<ExtractionResult, ExtractionError>
where
    F: FnOnce(
            tokio::runtime::Handle,
        ) -> Result<
            core_extraction::ExtractionResult,
            core_extraction::FileExtractionError,
        > + Send
        + 'static,
{
    let handle = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        crate::stack::run(move || extract(handle))
            .map_err(runtime_error)?
            .map_err(ExtractionError::from)
    })
    .await
    .map_err(runtime_error)??
    .try_into()
}

impl From<ExtractionOptions> for core_extraction::ExtractionConfig {
    fn from(value: ExtractionOptions) -> Self {
        Self {
            return_metadata: value.return_metadata,
            return_content: value.return_content,
            return_parts: value.return_parts,
            ocr: value.ocr.into(),
        }
    }
}

impl TryFrom<core_extraction::ExtractionResult> for ExtractionResult {
    type Error = ExtractionError;

    fn try_from(
        value: core_extraction::ExtractionResult,
    ) -> Result<Self, Self::Error> {
        Ok(Self {
            metadata: value.metadata.map(Into::into),
            pipeline: value
                .pipeline
                .into_iter()
                .map(ExtractionPipelineStep::try_from)
                .collect::<Result<Vec<_>, _>>()?,
            text: value.text,
            parts: value
                .parts
                .map(|parts| {
                    parts
                        .into_iter()
                        .map(ExtractionPart::try_from)
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?,
        })
    }
}

impl From<core_extraction::ExtractionMetadata> for ExtractionMetadata {
    fn from(value: core_extraction::ExtractionMetadata) -> Self {
        Self {
            stem: value.stem,
            extension: value.extension,
            label: value.label,
            mime_type: value.mime_type,
            description: value.description,
            is_text: value.is_text,
            confidence: value.confidence,
            origin: value.origin.into(),
            hash: value.hash,
        }
    }
}

impl TryFrom<core_extraction::ExtractionPart> for ExtractionPart {
    type Error = ExtractionError;

    fn try_from(
        value: core_extraction::ExtractionPart,
    ) -> Result<Self, Self::Error> {
        Ok(Self {
            index: u64::try_from(value.index).map_err(runtime_error)?,
            kind: value.kind.into(),
            text: value.text,
            provenance: value
                .provenance
                .map(ExtractionProvenance::try_from)
                .transpose()?,
        })
    }
}

impl TryFrom<core_extraction::ExtractionProvenance> for ExtractionProvenance {
    type Error = ExtractionError;

    fn try_from(
        value: core_extraction::ExtractionProvenance,
    ) -> Result<Self, Self::Error> {
        Ok(Self {
            confidence: value.confidence,
            page: value
                .page
                .map(u64::try_from)
                .transpose()
                .map_err(runtime_error)?,
            bbox: value.bbox.map(Into::into),
        })
    }
}

impl From<core_extraction::ExtractionBbox> for ExtractionBbox {
    fn from(value: core_extraction::ExtractionBbox) -> Self {
        Self {
            x: value.x,
            y: value.y,
            width: value.width,
            height: value.height,
        }
    }
}

impl TryFrom<core_extraction::ExtractionPipelineStep>
    for ExtractionPipelineStep
{
    type Error = ExtractionError;

    fn try_from(
        value: core_extraction::ExtractionPipelineStep,
    ) -> Result<Self, Self::Error> {
        Ok(Self {
            step: value.step.into(),
            engine: value.engine,
            duration_ms: value.duration_ms,
            outputs: value.outputs.into_iter().collect(),
        })
    }
}

impl From<core_extraction::ExtractionPipelineStepKind>
    for ExtractionPipelineStepKind
{
    fn from(value: core_extraction::ExtractionPipelineStepKind) -> Self {
        match value {
            core_extraction::ExtractionPipelineStepKind::Detection => {
                Self::Detection
            }
            core_extraction::ExtractionPipelineStepKind::Parsing => {
                Self::Parsing
            }
            core_extraction::ExtractionPipelineStepKind::Recognition => {
                Self::Recognition
            }
        }
    }
}

impl From<core_extraction::PartKind> for PartKind {
    fn from(value: core_extraction::PartKind) -> Self {
        match value {
            core_extraction::PartKind::Caption => Self::Caption,
            core_extraction::PartKind::Footer => Self::Footer,
            core_extraction::PartKind::Heading => Self::Heading,
            core_extraction::PartKind::ListItem => Self::ListItem,
            core_extraction::PartKind::Paragraph => Self::Paragraph,
            core_extraction::PartKind::Table => Self::Table,
            core_extraction::PartKind::Text => Self::Text,
            core_extraction::PartKind::Unknown => Self::Unknown,
        }
    }
}

impl From<core_extraction::FileExtractionError> for ExtractionError {
    fn from(value: core_extraction::FileExtractionError) -> Self {
        match value {
            core_extraction::FileExtractionError::Io { source } => Self::Io {
                message: source.to_string(),
            },
            error => Self::Runtime {
                message: error.to_string(),
            },
        }
    }
}

fn runtime_error(error: impl ToString) -> ExtractionError {
    ExtractionError::Runtime {
        message: error.to_string(),
    }
}
