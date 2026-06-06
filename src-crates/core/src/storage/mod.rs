//! Graph storage and retrieval built on `grafeo`.
//!
//! Provides the backend-neutral [`crate::storage::graph::GraphDbContext`] trait alongside
//! node, edge, and search types. Backend implementations are crate-private
//! and reached through [`crate::storage::graph::open_context`] or
//! [`crate::storage::graph::in_memory_context`].
//!
//! These items live in the `graph` submodule and are re-exported at the
//! `storage` root, so `akuna_core::storage::GraphNode` works the same as
//! `akuna_core::storage::graph::GraphNode`.
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
