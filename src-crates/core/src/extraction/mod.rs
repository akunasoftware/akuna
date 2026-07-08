//! File content and metadata extraction with structured parts.
//!
//! # Example
//!
//! ```rust,no_run
//! use akuna_core::extraction::{extract_file, ExtractionConfig};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let result = extract_file("path/to/file.pdf".as_ref(), &ExtractionConfig::default()).await?;
//! # Ok(())
//! # }
//! ```

mod document;
mod errors;
mod extractors;
mod metadata;
mod parts;
mod pipeline;
mod provenance;
mod types;

#[cfg(test)]
mod tests;

pub use crate::chunking::PartKind;
pub use crate::detection::DetectionOrigin;
pub use document::{extract_bytes, extract_file};
pub use errors::FileExtractionError;
pub(in crate::extraction) use types::DocumentContent;
pub use types::{
    ExtractionBbox, ExtractionByteRange, ExtractionConfig, ExtractionMetadata,
    ExtractionPart, ExtractionPipelineStep, ExtractionPipelineStepKind,
    ExtractionProvenance, ExtractionResult,
};
