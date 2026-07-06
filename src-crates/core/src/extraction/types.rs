use std::collections::HashMap;

use serde::Serialize;

/// Top-level extraction configuration.
pub struct ExtractionConfig {
    /// Include inferred file metadata in the result.
    pub return_metadata: bool,
    /// Include extracted content in the result.
    pub return_content: bool,
    /// Include structured content parts in the result.
    pub return_parts: bool,
    /// OCR configuration for image extraction.
    #[cfg(feature = "ocr")]
    pub ocr: crate::ocr::OcrEngineOptions,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            return_metadata: true,
            return_content: false,
            return_parts: false,
            #[cfg(feature = "ocr")]
            ocr: crate::ocr::OcrEngineOptions::default(),
        }
    }
}

/// Structured extraction output.
#[derive(Debug, Serialize)]
pub struct ExtractionResult {
    /// Detected file metadata, when requested
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ExtractionMetadata>,
    /// Processing steps applied to this document, when available.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub pipeline: Vec<ExtractionPipelineStep>,
    /// Extracted text, when requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Structured content parts derived from extraction, when content was read.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parts: Option<Vec<ExtractionPart>>,
}

/// Closed set of extraction pipeline step roles.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionPipelineStepKind {
    /// File type detection.
    Detection,
    /// Structured or plain content parsing.
    Parsing,
    /// Text recognition from image regions.
    Recognition,
}

/// Structured content part derived from a source document.
#[derive(Clone, Debug, Serialize)]
pub struct ExtractionPart {
    /// Zero-based part index.
    pub index: usize,
    /// Semantic part kind.
    pub kind: PartKind,
    /// Extracted part text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Source location and extractor details for this part, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<ExtractionProvenance>,
}

/// Source location details for an extracted part.
#[derive(Clone, Debug, Serialize)]
pub struct ExtractionProvenance {
    /// Recognition confidence from 0 to 1, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// One-based page number, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<usize>,
    /// Bounding box in source coordinate space, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox: Option<ExtractionBbox>,
    /// Byte range in the source text, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_range: Option<ExtractionByteRange>,
}

/// Shared semantic kind for extracted parts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PartKind {
    /// Caption or source attribution.
    Caption,
    /// Source code or code-like content.
    Code,
    /// Page footer content.
    Footer,
    /// Heading or title content.
    Heading,
    /// List item content.
    ListItem,
    /// Markup content not otherwise classified.
    Markup,
    /// Paragraph text.
    Paragraph,
    /// Table content.
    Table,
    /// Plain text content.
    Text,
    /// Unclassified content.
    Unknown,
}

/// One step in the extraction pipeline that processed a document.
#[derive(Clone, Debug, Serialize)]
pub struct ExtractionPipelineStep {
    /// Step role.
    pub step: ExtractionPipelineStepKind,
    /// Engine that performed this step (e.g. model identifier, library name).
    pub engine: String,
    /// Wall-clock duration of this step in milliseconds.
    pub duration_ms: u64,
    /// Throughput metrics.
    ///
    /// Known keys:
    ///
    /// - `pages`: pages parsed.
    /// - `parts`: structured parts emitted.
    /// - `texts`: recognized text blocks.
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub outputs: HashMap<String, u64>,
}

/// Bounding box in source coordinates.
#[derive(Clone, Debug, Serialize)]
pub struct ExtractionBbox {
    /// Left coordinate.
    pub x: f32,
    /// Top coordinate.
    pub y: f32,
    /// Box width.
    pub width: f32,
    /// Box height.
    pub height: f32,
}

/// Byte range in source content.
#[derive(Clone, Debug, Serialize)]
pub struct ExtractionByteRange {
    /// Inclusive start byte offset.
    pub start: usize,
    /// Exclusive end byte offset.
    pub end: usize,
}

/// Metadata detected during extraction.
#[derive(Clone, Debug, Serialize)]
pub struct ExtractionMetadata {
    /// File stem from the path, without the extension, when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stem: Option<String>,
    /// File extension from the path, when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extension: Option<String>,
    /// Unique label identifying the detected content type.
    pub label: String,
    /// Detected MIME type.
    pub mime_type: String,
    /// Detected file type description.
    pub description: String,
    /// Whether the file can be treated as text.
    pub is_text: bool,
    /// Blake3 hash of raw file bytes.
    pub hash: String,
}

impl std::fmt::Display for ExtractionMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let json =
            serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

/// Internal normalized document content.
pub(in crate::extraction) struct DocumentContent {
    pub(in crate::extraction) canonical_text: Option<String>,
    pub(in crate::extraction) parts: Vec<ExtractionPart>,
    pub(in crate::extraction) pipeline: Vec<ExtractionPipelineStep>,
}

impl DocumentContent {
    /// Build a plain text document part from extractor text.
    pub(in crate::extraction) fn from_text(text: String) -> Self {
        let parts = crate::extraction::parts::from_text(&text);
        if parts.len() > 1 {
            return Self {
                canonical_text: Some(text),
                parts,
                pipeline: Vec::new(),
            };
        }

        Self {
            canonical_text: None,
            parts: vec![ExtractionPart {
                index: 0,
                kind: PartKind::Text,
                text: Some(text),
                provenance: None,
            }],
            pipeline: Vec::new(),
        }
    }

    /// Returns canonical text by joining text-bearing parts in order.
    pub(in crate::extraction) fn text(&self) -> Option<String> {
        if let Some(text) = self.canonical_text.clone() {
            return Some(text);
        }

        let text = self
            .parts
            .iter()
            .filter_map(|part| part.text.as_deref())
            .collect::<Vec<_>>()
            .join("\n\n");

        (!text.is_empty()).then_some(text)
    }
}
