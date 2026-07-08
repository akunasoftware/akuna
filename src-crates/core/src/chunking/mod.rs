//! Text segmentation and retrieval chunking.
//!
//! ```
//! use akuna_core::chunking::ChunkingOptions;
//!
//! assert!(ChunkingOptions::default().enabled);
//! ```

// Shared internals are intentionally unused until extraction/index select them.
#![allow(dead_code)]

pub(crate) mod code;
mod errors;
pub(crate) mod packer;
pub(crate) mod prose;

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};

pub use errors::ChunkingError;

/// Options controlling how record content is split for retrieval.
#[derive(
    Clone, Debug, PartialEq, Eq, Deserialize, Serialize, utoipa::ToSchema,
)]
pub struct ChunkingOptions {
    /// Enable retrieval-sized splitting.
    pub enabled: bool,
    /// Maximum characters in each chunk.
    pub max_chars: usize,
    /// Characters repeated inside split segments.
    pub overlap_chars: usize,
}

impl Default for ChunkingOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            max_chars: 1600,
            overlap_chars: 200,
        }
    }
}

impl ChunkingOptions {
    /// Validates the configured chunk size and overlap.
    pub fn validate(&self) -> Result<(), ChunkingError> {
        if self.max_chars == 0 {
            return Err(ChunkingError::ZeroMaxChars);
        }
        if self.overlap_chars >= self.max_chars {
            return Err(ChunkingError::OverlapTooLarge {
                overlap_chars: self.overlap_chars,
                max_chars: self.max_chars,
            });
        }

        Ok(())
    }
}

/// Shared semantic kind for extracted content.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
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

/// Source segment used by extraction and chunk packing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Segment<'a> {
    pub(crate) text: &'a str,
    pub(crate) kind: PartKind,
    pub(crate) byte_range: std::ops::Range<usize>,
}
