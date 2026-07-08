use std::fmt;

type VectorErrorSource = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Vector operation target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VectorTarget {
    /// Vector table identified by name.
    Table {
        /// Table name.
        name: &'static str,
    },
    /// Record row identified by collection and record ID.
    Record {
        /// Collection containing the record.
        collection: String,
        /// Stable record identifier within the collection.
        record_id: String,
    },
    /// Chunk rows identified by collection and record ID.
    Chunks {
        /// Collection containing the record.
        collection: String,
        /// Stable record identifier within the collection.
        record_id: String,
    },
}

impl fmt::Display for VectorTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Table { name } => write!(formatter, "table '{name}'"),
            Self::Record {
                collection,
                record_id,
            } => write!(
                formatter,
                "record '{record_id}' in collection '{collection}'",
            ),
            Self::Chunks {
                collection,
                record_id,
            } => write!(
                formatter,
                "chunks for record '{record_id}' in collection '{collection}'",
            ),
        }
    }
}

/// Full-text search index target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorTextIndex {
    /// Chunk text index.
    Chunks,
    /// Title text index.
    Titles,
}

impl fmt::Display for VectorTextIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Chunks => formatter.write_str("chunk text"),
            Self::Titles => formatter.write_str("title text"),
        }
    }
}

/// Vector write operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorWriteOperation {
    /// Store or update vector data.
    Put,
    /// Delete vector data.
    Delete,
}

impl fmt::Display for VectorWriteOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Put => formatter.write_str("put"),
            Self::Delete => formatter.write_str("delete"),
        }
    }
}

/// Errors that can occur during vector operations.
#[derive(Debug, thiserror::Error)]
pub enum VectorError {
    /// Database engine failed to initialise.
    #[error("Database engine '{engine}' failed to initialise")]
    DbInit {
        /// The name of the engine that failed.
        engine: &'static str,
        /// The underlying database error.
        source: VectorErrorSource,
    },

    /// Error converting domain data into storage rows.
    #[error("Vector serialization failed")]
    Serialization {
        /// The underlying serialization error.
        source: VectorErrorSource,
    },

    /// Error converting storage rows into domain data.
    #[error("Vector deserialization failed")]
    Deserialization {
        /// The underlying deserialization error.
        source: VectorErrorSource,
    },

    /// Embedding dimensions are invalid.
    #[error("Embedding dimensions must be greater than zero and fit in i32")]
    InvalidDimensions {
        /// Invalid embedding dimension count.
        dimensions: usize,
    },

    /// Embedding dimensions did not match the context.
    #[error("Embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected embedding dimension count.
        expected: usize,
        /// Actual embedding dimension count.
        actual: usize,
    },

    /// Chunk row identity disagreed with the write target.
    #[error("Chunk '{chunk_id}' does not belong to target record")]
    InvalidChunkEntry {
        /// Target collection.
        expected_collection: String,
        /// Target record identifier.
        expected_record_id: String,
        /// Invalid chunk identifier.
        chunk_id: String,
        /// Chunk collection.
        actual_collection: String,
        /// Chunk record identifier.
        actual_record_id: String,
    },

    /// Error mutating vector data.
    #[error("Vector {operation} failed for {target} using engine '{engine}'")]
    WriteFailed {
        /// The name of the engine that failed.
        engine: &'static str,
        /// The write operation that failed.
        operation: VectorWriteOperation,
        /// The write target that failed.
        target: VectorTarget,
        /// The underlying database error.
        source: VectorErrorSource,
    },

    /// Error executing a vector query.
    #[error("Vector query execution failed with engine '{engine}'")]
    QueryExecution {
        /// The name of the engine that failed.
        engine: &'static str,
        /// The underlying database error.
        source: VectorErrorSource,
    },

    /// Full-text search was not enabled for the context.
    #[error("Full-text search is disabled for {target}")]
    FullTextDisabled {
        /// Full-text index target.
        target: VectorTextIndex,
    },
}
