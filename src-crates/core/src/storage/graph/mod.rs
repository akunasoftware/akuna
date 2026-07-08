//! Graph storage domain types and traits.
//!
//! ```no_run
//! use akuna_core::storage::graph::in_memory_context;
//!
//! let _ = in_memory_context();
//! ```

mod backend;
mod error;

use serde::{Deserialize, Serialize};

pub use error::{GraphError, GraphTarget, GraphWriteOperation};

/// Flexible relationship between knowledge graph nodes.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct GraphEdge {
    /// Source node labels.
    pub source_labels: Vec<String>,
    /// Stable source node identifier within its labels.
    pub source: String,
    /// Relationship type from source to target.
    pub predicate: String,
    /// Stable target node identifier within its labels.
    pub target: String,
    /// Target node labels.
    pub target_labels: Vec<String>,
}

/// Flexible knowledge graph concept with caller-defined labels and metadata.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct GraphNode {
    /// Stable concept identifier within its labels.
    pub id: String,
    /// Graph labels for this concept.
    pub labels: Vec<String>,
    /// Human-readable concept name.
    pub name: String,
    /// Optional concept description.
    pub description: Option<String>,
    /// Serializable concept metadata.
    pub metadata: Option<serde_json::Value>,
}

/// Typed graph storage context.
pub trait GraphDbContext: Send + Sync {
    /// Stores a graph node.
    fn put_node(&self, node: &GraphNode) -> Result<(), GraphError>;

    /// Reads a graph node by id and labels, if it exists.
    fn get_node(
        &self,
        labels: &[&str],
        id: &str,
    ) -> Result<Option<GraphNode>, GraphError>;

    /// Deletes a graph node by id and labels.
    fn delete_node(&self, labels: &[&str], id: &str) -> Result<(), GraphError>;

    /// Stores a graph edge between existing node ids.
    fn put_edge(&self, edge: &GraphEdge) -> Result<(), GraphError>;

    /// Deletes a graph edge by its identity.
    fn delete_edge(&self, edge: &GraphEdge) -> Result<(), GraphError>;

    /// Lists nodes connected to a graph node.
    fn neighbors(
        &self,
        labels: &[&str],
        id: &str,
    ) -> Result<Vec<(GraphEdge, GraphNode)>, GraphError>;
}

/// Opens a persistent graph storage context rooted at `path`.
pub fn open_context(
    path: impl AsRef<std::path::Path>,
) -> Result<Box<dyn GraphDbContext>, GraphError> {
    Ok(Box::new(backend::GrafeoDbContext::new(
        path.as_ref().to_path_buf(),
    )?))
}

/// Creates an in-memory graph storage context for tests and ephemeral use.
pub fn in_memory_context() -> Box<dyn GraphDbContext> {
    Box::new(backend::GrafeoDbContext::new_in_memory())
}

#[cfg(test)]
mod tests;
