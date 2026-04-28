//! Extraction result and configuration types.

mod errors;

use std::collections::HashMap;

use serde::Serialize;

pub use errors::*;

/// Top-level extraction configuration.
/// File read will only happen if content or chunk return are enabled
/// Metadata extract will not result in file extract, only metadata inference
pub struct ExtractionConfig {
    /// Whether inferred file metadata should be included in extraction output
    pub return_metadata: bool,
    /// Whether content should be included in extraction output
    /// If false but chunking enabled, content will still be extracted from file but not returned
    pub return_content: bool,
    /// Whether chunks should be included in extraction output
    /// If false, chunking config will have no effect
    pub return_chunking: bool,
    /// Optional preferences for text content behaviour
    /// Only effective if return_content or return_chunking are enabled
    pub text: Option<TextExtractionConfig>,
    /// Optional preferences for chunking behaviour
    /// Only effective if return_chunking is enabled
    pub chunking: Option<ChunkingConfig>,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            return_metadata: true,
            return_content: false,
            return_chunking: false,
            text: Some(TextExtractionConfig::default()),
            chunking: Some(ChunkingConfig::default()),
        }
    }
}

/// Preferences for how text should be extracted
#[derive(Default)]
pub struct TextExtractionConfig {
    /// Whether supported extractors should prefer Markdown output.
    pub prefer_markdown: bool,
}

/// Text chunking configuration.
#[derive(Default)]
pub struct ChunkingConfig {
    /// Optional target chunk size in bytes.
    pub target_size: Option<usize>,
    /// Optional delimiter bytes keyed by detected file type label or MIME type.
    pub delimiters_by_ft: HashMap<String, Vec<u8>>,
    /// Optional delimiter bytes for chunk boundary selection.
    pub delimiters: Option<Vec<u8>>,
}

/// Structured extraction output.
#[derive(Debug, Serialize)]
pub struct ExtractionResult {
    /// Detected file metadata, when requested
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ExtractionMetadata>,
    /// Extracted text content, when requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ExtractionContent>,
    /// Extracted text chunks, when requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunks: Option<Vec<ExtractionChunk>>,
}

/// Extracted text content and related derived data.
#[derive(Debug, Serialize)]
pub struct ExtractionContent {
    /// Extracted text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Extracted text chunk and related derived data.
#[derive(Debug, Serialize)]
pub struct ExtractionChunk {
    /// Zero-based chunk index.
    pub index: usize,
    /// Chunk text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Metadata detected during extraction.
#[derive(Debug, Serialize)]
pub struct ExtractionMetadata {
    /// File stem from the path, without the extension, when present.
    pub stem: Option<String>,
    /// File extension from the path, when present.
    pub extension: Option<String>,
    /// Unique label identifying the detected content type.
    pub label: String,
    /// Detected MIME type.
    pub mime_type: String,
    /// Detected file type description.
    pub description: String,
}

/// File type data detected before extraction.
#[derive(Debug)]
pub(crate) struct DetectedFileType {
    /// Detected MIME type.
    pub(crate) mime_type: String,
    /// Unique label identifying the detected content type.
    pub(crate) label: String,
    /// Detected file type description.
    pub(crate) description: String,
    /// Whether the file can be treated as text.
    pub(crate) is_text: bool,
}
