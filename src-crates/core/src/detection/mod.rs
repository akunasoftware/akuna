//! File-type detection.
//!
//! Classifies raw bytes or files into typed labels with confidence scores.
//!
//! # Example
//!
//! ```rust,no_run
//! use akuna_core::detection::FileTypeDetector;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let detector = FileTypeDetector::new()?;
//!
//!     let detected = detector.identify_bytes(b"fn main() { println!(\"hi\"); }")?;
//!     println!("{} {}", detected.info().label, detected.info().mime_type);
//!
//!     Ok(())
//! }
//! ```

mod config;
mod detector;
mod errors;
mod models;
mod types;
mod vendor;

#[cfg(test)]
mod tests;

pub use detector::FileTypeDetector;
pub use errors::DetectionError;
pub use types::{DetectionOrigin, FileType, FileTypeInfo};
