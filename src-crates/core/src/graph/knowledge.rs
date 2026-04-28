use serde::{Deserialize, Serialize};

use crate::graph::primitives::{GraphEdge, GraphNode};

/// Flexible relationship between knowledge graph nodes.
#[derive(Clone, Debug, GraphEdge, PartialEq, Eq, Deserialize, Serialize)]
pub struct Relationship {
    /// Source node labels.
    #[graph(source_labels)]
    pub source_labels: Vec<String>,
    /// Stable source node identifier within its labels.
    #[graph(source)]
    pub source: String,
    /// Relationship type from source to target.
    #[graph(predicate)]
    pub predicate: String,
    /// Stable target node identifier within its labels.
    #[graph(target)]
    pub target: String,
    /// Target node labels.
    #[graph(target_labels)]
    pub target_labels: Vec<String>,
}

/// Flexible knowledge graph concept with caller-defined labels and metadata.
#[derive(Clone, Debug, GraphNode, PartialEq, Deserialize, Serialize)]
pub struct Concept {
    /// Stable concept identifier within its labels.
    #[graph(id)]
    pub id: String,
    /// Graph labels for this concept.
    #[graph(labels)]
    pub labels: Vec<String>,
    /// Human-readable concept name.
    #[graph(name)]
    pub name: String,
    /// Optional concept description.
    #[graph(description)]
    pub description: Option<String>,
    /// Serializable concept metadata.
    #[graph(metadata)]
    pub metadata: Option<serde_json::Value>,
}

/// Opinionated knowledge graph node for representing a claim.
#[derive(Clone, Debug, GraphNode, PartialEq, Deserialize, Serialize)]
#[graph(node_type(
    name = "Assertion",
    description = "Claim or statement represented in the knowledge graph"
))]
pub struct Assertion {
    /// Stable assertion identifier within its labels.
    #[graph(id)]
    pub id: String,
    /// Serializable assertion metadata.
    #[graph(metadata)]
    pub metadata: Option<serde_json::Value>,
    // TODO: WIP
}

/// Opinionated knowledge graph node for representing source context.
#[derive(Clone, Debug, GraphNode, PartialEq, Deserialize, Serialize)]
#[graph(node_type(
    name = "Provenance",
    description = "Source context for knowledge graph assertions or entities"
))]
pub struct Provenance {
    /// Stable provenance identifier within its labels.
    #[graph(id)]
    pub id: String,
    /// Serializable provenance metadata.
    #[graph(metadata)]
    pub metadata: Option<serde_json::Value>,
    // TODO: WIP
}
