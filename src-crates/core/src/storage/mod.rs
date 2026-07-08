//! Storage primitives and engines.
//!
//! ```no_run
//! use akuna_core::storage::vector::{VectorContextOptions, in_memory_context};
//!
//! # async fn example() -> Result<(), akuna_core::storage::VectorError> {
//! let _ = in_memory_context(384, &VectorContextOptions::default()).await?;
//! # Ok(())
//! # }
//! ```

pub mod graph;
pub mod vector;

pub use graph::{
    GraphDbContext, GraphEdge, GraphError, GraphNode, GraphTarget,
    GraphWriteOperation, in_memory_context, open_context,
};
pub use vector::{
    ChunkEntry, ChunkSearchResult, RecordEntry, RecordSearchResult,
    VectorDbContext, VectorError, VectorSearchQuery, VectorTarget,
    VectorWriteOperation,
};
