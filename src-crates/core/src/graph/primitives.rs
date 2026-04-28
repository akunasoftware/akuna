//! Graph primitives.

use serde::{Serialize, de::DeserializeOwned};

use crate::GraphError;

pub use akuna_core_macros::{GraphEdge, GraphNode};

use crate::graph::storage;

/// Domain object that can be stored as a graph node.
pub trait GraphNode {
    /// Serializable graph node metadata.
    type Metadata: Serialize + DeserializeOwned;

    /// Graph node labels used by storage backends.
    fn labels(&self) -> Vec<&str>;

    /// Stable graph node identifier within its labels.
    fn id(&self) -> &str;

    /// Human-readable node name.
    fn name(&self) -> &str;

    /// Optional node description.
    fn description(&self) -> Option<&str>;

    /// Serializable metadata of domain object.
    fn metadata(&self) -> Option<&Self::Metadata>;

    /// Builds a node from graph storage parts.
    fn from_graph_parts(
        id: String,
        labels: Vec<String>,
        name: String,
        description: Option<String>,
        metadata: Option<Self::Metadata>,
    ) -> Self;
}

/// Domain relationship that can be stored as a graph edge.
pub trait GraphEdge {
    /// Source node labels.
    fn source_labels(&self) -> Vec<&str>;

    /// Stable source node identifier within its labels.
    fn source(&self) -> &str;

    /// Relationship type from source to target.
    fn predicate(&self) -> &str;

    /// Stable target node identifier within its labels.
    fn target(&self) -> &str;

    /// Target node labels.
    fn target_labels(&self) -> Vec<&str>;
}

impl<T> GraphEdge for &T
where
    T: GraphEdge + ?Sized,
{
    fn source_labels(&self) -> Vec<&str> {
        T::source_labels(self)
    }

    fn source(&self) -> &str {
        T::source(self)
    }

    fn predicate(&self) -> &str {
        T::predicate(self)
    }

    fn target(&self) -> &str {
        T::target(self)
    }

    fn target_labels(&self) -> Vec<&str> {
        T::target_labels(self)
    }
}

/// Typed graph storage context.
pub trait GraphDbContext {
    /// Storage mode backing this graph context.
    fn storage(&self) -> &storage::GraphStorage;

    /// Stores a graph node from domain data.
    fn put_node<T>(&self, node: &T) -> Result<(), GraphError>
    where
        T: GraphNode + ?Sized;

    /// Reads a graph node by domain id.
    fn get_node<T>(
        &self,
        labels: &[&str],
        id: impl AsRef<str>,
    ) -> Result<Option<T>, GraphError>
    where
        T: GraphNode;

    /// Deletes an existing graph node by domain id.
    fn delete_node<T>(
        &self,
        labels: &[&str],
        id: impl AsRef<str>,
    ) -> Result<(), GraphError>
    where
        T: GraphNode;

    /// Stores a graph edge between existing node ids.
    fn put_edge<T>(&self, edge: &T) -> Result<(), GraphError>
    where
        T: GraphEdge + ?Sized;

    /// Deletes an existing graph edge by structural identity.
    fn delete_edge<T>(&self, edge: &T) -> Result<(), GraphError>
    where
        T: GraphEdge + ?Sized;

    /// Consumes the context and removes any persisted storage it owns.
    fn destroy(self) -> Result<(), GraphError>
    where
        Self: Sized;
}
