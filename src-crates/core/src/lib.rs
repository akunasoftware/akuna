//! Akuna core knowledge tooling library.
//!
//! Provides optional feature-gated modules for file-type detection,
//! embeddings, extraction, layout, OCR, reranking, and graph storage.
//! Consumers enable only the features they need.
//!
//! # Modules
//!
//! - [`detection`] — file-type detection (feature `detection`)
//! - [`embedding`] — text embeddings (feature `embedding`)
//! - [`extraction`] — file extraction (feature `extraction`)
//! - [`layout`] — document layout detection (feature `layout`)
//! - [`ocr`] — image OCR engines (feature `ocr`)
//! - [`reranking`] — text reranking (feature `reranking`)
//! - [`storage`] — graph storage and retrieval (feature `storage`)
//!
//! # Example
//!
//! Enable the `extraction` feature and call a module function:
//!
//! ```ignore
//! use akuna_core::extraction::{document, ExtractionConfig};
//!
//! # async fn example() -> Result<(), akuna_core::extraction::FileExtractionError> {
//! let result = document::from_path("path/to/file.pdf".as_ref(), &ExtractionConfig::default()).await?;
//! # Ok(())
//! # }
//! ```

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
