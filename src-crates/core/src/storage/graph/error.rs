use std::fmt;

type GraphErrorSource = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Graph operation target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphTarget {
    /// Graph node identified by labels and domain ID.
    Node {
        /// Graph node labels searched.
        labels: Vec<String>,
        /// Domain ID searched.
        id: String,
    },
    /// Graph edge identified by relationship type and endpoint IDs.
    Edge {
        /// Relationship type from source to target.
        predicate: String,
        /// Stable domain identifier for the source node.
        source_id: String,
        /// Stable domain identifier for the target node.
        target_id: String,
    },
}

impl fmt::Display for GraphTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Node { labels, id } => {
                write!(formatter, "node '{id}' with labels '{labels:?}'")
            }
            Self::Edge {
                predicate,
                source_id,
                target_id,
            } => write!(
                formatter,
                "edge '{predicate}' from '{source_id}' to '{target_id}'",
            ),
        }
    }
}

/// Graph write operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphWriteOperation {
    /// Store or update graph data.
    Put,
    /// Delete graph data.
    Delete,
}

impl fmt::Display for GraphWriteOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Put => formatter.write_str("put"),
            Self::Delete => formatter.write_str("delete"),
        }
    }
}

/// Errors that can occur during graph operations.
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    /// Database engine failed to initialise.
    #[error("Database engine '{engine}' failed to initialise")]
    DbInit {
        /// The name of the engine that failed.
        engine: &'static str,
        /// The underlying database error.
        source: GraphErrorSource,
    },

    /// Error converting domain data into graph properties.
    #[error("Graph serialization failed")]
    Serialization {
        /// The underlying serialization error.
        source: GraphErrorSource,
    },

    /// Error converting graph properties back into domain data.
    #[error("Graph deserialization failed")]
    Deserialization {
        /// The underlying deserialization error.
        source: GraphErrorSource,
    },

    /// Node properties were valid JSON, but not a graph property object.
    #[error("Node properties must serialize to an object")]
    InvalidNodeItem,

    /// Edge predicate cannot be used as a relationship type.
    #[error("Edge predicate '{predicate}' is not a valid relationship type")]
    InvalidEdgePredicate {
        /// Invalid edge predicate.
        predicate: String,
    },

    /// Graph target did not exist.
    #[error("Graph {target} does not exist")]
    NotFound {
        /// Graph target expected to exist.
        target: GraphTarget,
    },

    /// Error mutating graph data.
    #[error("Graph {operation} failed for {target} using engine '{engine}'")]
    WriteFailed {
        /// The name of the engine that failed.
        engine: &'static str,
        /// The write operation that failed.
        operation: GraphWriteOperation,
        /// The write target that failed.
        target: GraphTarget,
        /// The underlying database error.
        source: GraphErrorSource,
    },

    /// Error executing a graph query.
    #[error("Query execution failed with engine '{engine}'")]
    QueryExecution {
        /// The name of the engine that failed.
        engine: &'static str,
        /// The underlying database error.
        source: GraphErrorSource,
    },
}
