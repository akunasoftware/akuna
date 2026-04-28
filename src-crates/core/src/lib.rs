//! Shared public API's for common actions across the library.

extern crate self as akuna_core;

/// Text chunking APIs.
#[cfg(feature = "chunking")]
pub mod chunking;

/// Platform-aware application directories.
pub mod dirs;

/// File extraction APIs.
#[cfg(feature = "extraction")]
pub mod extraction;

/// Internal testing utilities.
#[cfg(any(test, feature = "testing"))]
pub mod testing;

/// Storage indexing & retrieval APIs.
#[cfg(feature = "graph")]
pub mod graph;

/// Application tracing helpers.
pub mod tracing;

/// Shared public types.
#[cfg(any(feature = "chunking", feature = "extraction", feature = "graph"))]
pub mod types;

#[cfg(any(feature = "chunking", feature = "extraction"))]
pub use types::extraction::*;

#[cfg(feature = "graph")]
pub use types::graph::*;

/// Text embeddings
#[cfg(feature = "embedding")]
pub mod embedding;

/// The name of the application.
/// Used for directory names, trace logs, etc.
pub const APP_NAME: &str = "akuna";
