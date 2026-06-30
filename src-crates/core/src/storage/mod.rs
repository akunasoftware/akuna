//! Graph storage and retrieval.
//!
//! Open a context with [`crate::storage::graph::open_context`] or
//! [`crate::storage::graph::in_memory_context`] and work through the
//! [`crate::storage::graph::GraphDbContext`] trait and its node, edge, and
//! search types. Items are re-exported at the `storage` root.
//!
//! # Example
//!
//! ```rust,no_run
//! use akuna_core::storage::graph::{
//!     in_memory_context, GraphDbContext, GraphNode,
//! };
//!
//! let ctx = in_memory_context();
//! let node = GraphNode {
//!     id: "rust".to_string(),
//!     labels: vec!["Concept".to_string()],
//!     name: "Rust".to_string(),
//!     description: None,
//!     metadata: None,
//! };
//! ctx.put_node(&node, &[]).expect("node stored");
//! ```

pub mod graph;

pub use graph::{
    GraphDbContext, GraphEdge, GraphError, GraphNode, GraphNodeSearchQuery,
    GraphNodeSearchResult, GraphTarget, GraphWriteOperation, in_memory_context,
    open_context, search_text,
};
