//! UniFFI bindings crate.
//!
//! ```rust,no_run
//! let _detector = akuna_ffi::detection::FileTypeDetector::new()?;
//! # Ok::<(), akuna_ffi::detection::DetectionError>(())
//! ```

pub mod detection;
pub mod embedding;
pub mod extraction;
pub mod index;
pub mod layout;
pub mod ocr;
pub mod reranking;
mod stack;

uniffi::setup_scaffolding!("akuna_core");
