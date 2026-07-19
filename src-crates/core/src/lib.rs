//! Akuna core knowledge tooling library.
//!
//! Feature-gated capabilities for file-type detection, embeddings, extraction,
//! OCR, and reranking.
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "extraction")]
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use akuna_core::extraction::{extract_file, ExtractionConfig};
//! let result = extract_file("path/to/file.pdf".as_ref(), &ExtractionConfig::default()).await?;
//! # Ok(())
//! # }
//! ```

/// Package name used for platform integration.
pub const PACKAGE_NAME: &str = "akuna";

/// Shared crate-local test helpers.
#[cfg(all(test, any(feature = "detection", feature = "embedding")))]
mod testkit;

/// File-type detection APIs.
#[cfg(feature = "detection")]
pub mod detection;

/// Text embeddings.
#[cfg(feature = "embedding")]
pub mod embedding;

/// Text reranking APIs.
#[cfg(feature = "reranking")]
pub mod reranking;

/// Image OCR and document layout APIs.
#[cfg(feature = "ocr")]
pub mod ocr;

/// Shared ML model helpers.
#[cfg(feature = "ml")]
mod ml;

/// File extraction APIs.
#[cfg(feature = "extraction")]
pub mod extraction;
