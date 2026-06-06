//! File-type detection using Magika and Burn.
//!
//! Classifies raw bytes or files into typed labels with confidence scores.
//! The Magika weights are embedded in the binary, so no model download is
//! needed at runtime.
//!
//! # Models
//!
//! Backed by [`DetectionModel`][crate::detection::DetectionModel]:
//!
//! - `Magika` — Google Magika file-type classifier (weights embedded in the binary)
//!
//! # Example
//!
//! ```rust,no_run
//! use akuna_core::detection::Session;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut session = Session::new_default()?;
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

/// Supported file-type detection models.
///
/// Detection ships an embedded Magika model; see [`Session`] for loading.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DetectionModel {
    /// Google Magika file-type classifier (weights embedded in the binary).
    #[default]
    Magika,
}

/// One ranked label guess produced by the classifier.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedAlternative {
    /// Human-readable label for the candidate type.
    pub label: String,
    /// Optional MIME type associated with the label.
    pub mime_type: Option<String>,
    /// Model confidence score in `[0.0, 1.0]`.
    pub confidence: f32,
}

/// Top-level result of classifying a single input.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// Human-readable label of the most likely type.
    pub label: String,
    /// Optional MIME type of the most likely type.
    pub mime_type: Option<String>,
    /// Confidence score of the top prediction.
    pub confidence: f32,
    /// Alternative guesses ranked by confidence.
    pub alternatives: Vec<RankedAlternative>,
}

pub use config::ModelConfig;
pub use models::magika::MagikaInferenceError;
pub use session::Session;
pub use vendor::file::{FileType, InferredType, OverwriteReason, TypeInfo};
