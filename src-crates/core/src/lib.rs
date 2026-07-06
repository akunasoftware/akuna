//! Akuna core knowledge tooling library.
//!
//! Feature-gated modules for file-type detection, embeddings, extraction,
//! layout, OCR, reranking, and graph storage. Enable only the features you need.
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

/// Shared crate-local test helpers.
#[cfg(all(test, feature = "extraction"))]
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

/// Image OCR APIs.
#[cfg(feature = "ocr")]
pub mod ocr;

/// Document layout detection APIs.
#[cfg(feature = "layout")]
pub mod layout;

/// Shared ML model helpers.
#[cfg(feature = "ml")]
mod ml;

/// File extraction APIs.
#[cfg(feature = "extraction")]
pub mod extraction;

/// Graph storage and retrieval APIs.
#[cfg(feature = "storage")]
pub mod storage;
