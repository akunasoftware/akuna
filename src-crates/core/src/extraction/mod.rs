//! File content and metadata extraction with structured parts.
//!
//! Extraction reads supported files into structured parts and derived text.
//! File type detection is ML-backed when feature `detection` is enabled, and
//! extension-based otherwise.
//!
//! # Example
//!
//! ```rust,no_run
//! use akuna_core::extraction::{document, ExtractionConfig};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let result = document::from_path("path/to/file.pdf".as_ref(), &ExtractionConfig::default()).await?;
//! # Ok(())
//! # }
//! ```

pub mod document;
mod errors;
mod extractors;
mod metadata;
mod parts;
mod pipeline;
mod provenance;
#[cfg(test)]
mod tests;
mod types;

pub use errors::FileExtractionError;
pub(in crate::extraction) use types::DocumentContent;
pub use types::{
    ExtractionBbox, ExtractionByteRange, ExtractionConfig, ExtractionMetadata,
    ExtractionPart, ExtractionPipelineStep, ExtractionProvenance,
    ExtractionResult, PartKind,
};
