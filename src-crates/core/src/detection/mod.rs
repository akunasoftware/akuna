//! File-type detection.
//!
//! Classifies raw bytes or files into typed labels with confidence scores.
//!
//! # Example
//!
//! ```rust,no_run
//! use akuna_core::detection::Session;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let session = Session::new()?;
//!
//!     let detected = session.identify_content_sync(b"fn main() { println!(\"hi\"); }")?;
//!     println!("{} {}", detected.info().label, detected.info().mime_type);
//!
//!     Ok(())
//! }
//! ```

mod config;
mod models;
mod session;
mod vendor;

/// One ranked label guess produced by the classifier.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RankedAlternative {
    pub label: String,
    pub mime_type: Option<String>,
    pub confidence: f32,
}

/// Top-level result of classifying a single input.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Detection {
    pub label: String,
    pub mime_type: Option<String>,
    pub confidence: f32,
    pub alternatives: Vec<RankedAlternative>,
}

pub use models::magika::MagikaInferenceError;
pub use session::Session;
pub use vendor::file::{FileType, InferredType, OverwriteReason, TypeInfo};
