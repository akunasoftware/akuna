//! Graph storage domain types, traits, and backend wiring.
//!
//! Backend implementations live under `backend` (crate-private) and are
//! intentionally kept out of the public surface. Obtain a context via
//! [`open_context`] or [`in_memory_context`] and program against the
//! [`GraphDbContext`] trait.

mod backend;
mod error;

use serde::{Deserialize, Serialize};

pub use error::{GraphError, GraphTarget, GraphWriteOperation};

/// Flexible relationship between knowledge graph nodes.
#[derive(
    Clone, Debug, PartialEq, Eq, Deserialize, Serialize, utoipa::ToSchema,
)]
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
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, utoipa::ToSchema)]
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

/// Node search request sent to graph storage.
pub struct GraphNodeSearchQuery {
    /// Optional label to search within.
    pub label: Option<String>,
    /// Search text.
    pub query: String,
    /// Maximum result count.
    pub limit: usize,
}

/// Ranked graph node search result.
#[derive(Clone, Debug, PartialEq, Serialize, utoipa::ToSchema)]
pub struct GraphNodeSearchResult {
    /// Matching graph node.
    pub node: GraphNode,
    /// Fused relevance score.
    pub score: f64,
}

/// Typed graph storage context.
pub trait GraphDbContext: Send + Sync {
    /// Stores a graph node.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError`] on serialization failure, reserved property
    /// use, or backend write failure.
    fn put_node(
        &self,
        node: &GraphNode,
        search_embedding: &[f32],
    ) -> Result<(), GraphError>;

    /// Reads a graph node by id and labels.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError`] on backend query failure or if stored
    /// properties cannot be deserialized.
    fn get_node(
        &self,
        labels: &[&str],
        id: &str,
    ) -> Result<Option<GraphNode>, GraphError>;

    /// Deletes an existing graph node by id and labels.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::NotFound`] if no node matches, or [`GraphError`]
    /// on backend write failure.
    fn delete_node(&self, labels: &[&str], id: &str) -> Result<(), GraphError>;

    /// Searches graph nodes by hybrid text and vector relevance.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError`] on backend query failure or if results cannot
    /// be decoded.
    fn search_nodes(
        &self,
        query: &GraphNodeSearchQuery,
        query_embedding: &[f32],
    ) -> Result<Vec<GraphNodeSearchResult>, GraphError>;

    /// Stores a graph edge between existing node ids.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError`] on backend query or write failure.
    fn put_edge(&self, edge: &GraphEdge) -> Result<(), GraphError>;

    /// Deletes an existing graph edge by structural identity.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::NotFound`] if no edge matches, or [`GraphError`]
    /// on backend write failure.
    fn delete_edge(&self, edge: &GraphEdge) -> Result<(), GraphError>;
}

/// Builds the search text used to index a node for hybrid retrieval.
pub fn search_text(node: &GraphNode) -> String {
    let mut parts = vec![node.name.as_str()];
    if let Some(description) = node.description.as_deref() {
        parts.push(description);
    }

    parts.join("\n")
}

/// Opens a persistent graph storage context rooted at `path`.
///
/// Backend selection is currently fixed. Callers should depend only on the
/// returned [`GraphDbContext`] trait.
///
/// # Errors
///
/// Returns [`GraphError::DbInit`] if the backend fails to open at `path`.
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
