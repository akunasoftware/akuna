//! Index bindings.
//!
//! ```rust,no_run
//! # async fn example() -> Result<(), akuna_ffi::index::IndexError> {
//! let _index = akuna_ffi::index::load_index(None).await?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use akuna_core::chunking as core_chunking;
use akuna_core::index as core_index;

use crate::embedding::EmbeddingModel;
use crate::reranking::RerankingModel;

/// Index adapter error.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum IndexError {
    /// Index runtime failure.
    #[error("{message}")]
    Runtime {
        /// Human-readable error message.
        message: String,
    },
}

/// Options controlling how record content is split for retrieval.
#[derive(uniffi::Record)]
pub struct ChunkingOptions {
    /// Enable retrieval-sized splitting.
    pub enabled: bool,
    /// Maximum characters in each chunk.
    pub max_chars: u64,
    /// Characters repeated inside split segments.
    pub overlap_chars: u64,
}

/// Options for opening an [`Index`].
#[derive(uniffi::Record)]
pub struct IndexOptions {
    /// Storage subpath under the data root.
    pub name: String,
    /// Data root for persistent storage.
    pub path: Option<String>,
    /// Embedding model for records.
    pub embedding_model: EmbeddingModel,
    /// Reranking model for search ranking.
    pub reranking_model: Option<RerankingModel>,
    /// Enables lexical retrieval storage.
    pub fulltext: bool,
    /// Enables relationship graph storage.
    pub graph: bool,
    /// Chunking options for record content.
    pub chunking: ChunkingOptions,
    /// Hugging Face cache directory override.
    pub cache_dir: Option<String>,
}

/// Record metadata value.
#[derive(uniffi::Enum)]
pub enum MetadataValue {
    /// Text metadata value.
    Text {
        /// Metadata value.
        value: String,
    },
    /// Integer metadata value.
    Integer {
        /// Metadata value.
        value: i64,
    },
    /// Float metadata value.
    Float {
        /// Metadata value.
        value: f64,
    },
    /// Boolean metadata value.
    Boolean {
        /// Metadata value.
        value: bool,
    },
}

/// Record metadata predicate.
#[derive(uniffi::Enum)]
pub enum MetadataFilter {
    /// Metadata key must equal the supplied value.
    Equals {
        /// Metadata key to inspect.
        key: String,
        /// Metadata value to compare.
        value: MetadataValue,
    },
    /// Every metadata predicate must match.
    All {
        /// Metadata predicates to match.
        filters: Vec<MetadataFilter>,
    },
}

/// Relationship from one record to another.
#[derive(uniffi::Record)]
pub struct RecordRelationship {
    /// Relationship predicate.
    pub predicate: String,
    /// Target record id.
    pub record_id: String,
    /// Target collection.
    pub collection: String,
}

/// Record stored in an [`Index`].
#[derive(uniffi::Record)]
pub struct Record {
    /// Stable id within the collection.
    pub id: String,
    /// Collection containing the record.
    pub collection: String,
    /// Record title.
    pub title: String,
    /// Record content.
    pub content: String,
    /// Record metadata.
    pub metadata: HashMap<String, MetadataValue>,
    /// Outgoing record relationships.
    pub relationships: Vec<RecordRelationship>,
}

/// Search request for an [`Index`].
#[derive(uniffi::Record)]
pub struct IndexSearchQuery {
    /// Query text.
    pub text: String,
    /// Collections to search.
    pub collections: Vec<String>,
    /// Metadata predicate.
    pub filter: Option<MetadataFilter>,
    /// Maximum result count.
    pub limit: u64,
}

/// Search result from an [`Index`].
#[derive(uniffi::Record)]
pub struct IndexSearchResult {
    /// Matching record id.
    pub record_id: String,
    /// Matching collection.
    pub collection: String,
    /// Matching title.
    pub title: String,
    /// Matching metadata.
    pub metadata: HashMap<String, MetadataValue>,
    /// Relevance score.
    pub score: f32,
    /// Matching content preview.
    pub preview: Option<String>,
}

/// Index operation recorded for auditing.
#[derive(uniffi::Enum)]
pub enum IndexAuditOperation {
    /// Records were added or replaced.
    Add,
    /// A record was removed.
    Remove,
    /// Records were searched.
    Search,
}

/// Metered index operation record.
#[derive(uniffi::Record)]
pub struct IndexAuditRecord {
    /// Index operation that completed.
    pub operation: IndexAuditOperation,
    /// Engine that performed the operation.
    pub engine: String,
    /// Wall-clock operation duration in milliseconds.
    pub duration_ms: u64,
    /// Output counts keyed by output kind.
    pub outputs: HashMap<String, u64>,
}

/// Embedded retrieval index.
#[derive(uniffi::Object)]
pub struct Index {
    inner: core_index::Index,
}

#[uniffi::export(async_runtime = "tokio")]
/// Opens an index.
pub async fn load_index(
    options: Option<IndexOptions>,
) -> Result<Index, IndexError> {
    let inner =
        crate::stack::run_async(core_index::Index::new(core_options(options)?))
            .map_err(to_error)?
            .map_err(to_error)?;
    Ok(Index { inner })
}

#[uniffi::export(async_runtime = "tokio")]
impl Index {
    /// Adds or replaces records.
    pub async fn add(&self, records: Vec<Record>) -> Result<(), IndexError> {
        crate::stack::run_async(
            self.inner
                .add(records.into_iter().map(core_record).collect()),
        )
        .map_err(to_error)?
        .map_err(to_error)
    }

    /// Removes one record.
    pub async fn remove(
        &self,
        collection: String,
        record_id: String,
    ) -> Result<(), IndexError> {
        crate::stack::run_async(self.inner.remove(&collection, &record_id))
            .map_err(to_error)?
            .map_err(to_error)
    }

    /// Reads one record.
    pub async fn get(
        &self,
        collection: String,
        record_id: String,
    ) -> Result<Option<Record>, IndexError> {
        crate::stack::run_async(self.inner.get(&collection, &record_id))
            .map_err(to_error)?
            .map(|record| record.map(ffi_record))
            .map_err(to_error)
    }

    /// Searches records.
    pub async fn search(
        &self,
        query: IndexSearchQuery,
    ) -> Result<Vec<IndexSearchResult>, IndexError> {
        crate::stack::run_async(self.inner.search(core_search_query(query)?))
            .map_err(to_error)?
            .map(|results| results.into_iter().map(ffi_search_result).collect())
            .map_err(to_error)
    }

    /// Returns completed index operation audit records.
    pub fn audit_records(&self) -> Vec<IndexAuditRecord> {
        self.inner
            .audit_records()
            .into_iter()
            .map(Into::into)
            .collect()
    }
}

/// Converts FFI options to core options.
fn core_options(
    options: Option<IndexOptions>,
) -> Result<core_index::IndexOptions, IndexError> {
    let Some(options) = options else {
        return Ok(core_index::IndexOptions::default());
    };

    Ok(core_index::IndexOptions {
        name: options.name,
        path: options.path.map(PathBuf::from),
        embedding_model: options.embedding_model.into(),
        reranking_model: options.reranking_model.map(Into::into),
        fulltext: options.fulltext,
        graph: options.graph,
        chunking: core_chunking_options(options.chunking)?,
        cache_dir: options.cache_dir.map(PathBuf::from),
    })
}

/// Converts FFI chunking options to core options.
fn core_chunking_options(
    options: ChunkingOptions,
) -> Result<core_chunking::ChunkingOptions, IndexError> {
    Ok(core_chunking::ChunkingOptions {
        enabled: options.enabled,
        max_chars: usize::try_from(options.max_chars).map_err(to_error)?,
        overlap_chars: usize::try_from(options.overlap_chars)
            .map_err(to_error)?,
    })
}

/// Converts an FFI record to a core record.
fn core_record(record: Record) -> core_index::Record {
    core_index::Record {
        id: record.id,
        collection: record.collection,
        title: record.title,
        content: record.content,
        metadata: record
            .metadata
            .into_iter()
            .map(|(key, value)| (key, value.into()))
            .collect(),
        relationships: record
            .relationships
            .into_iter()
            .map(|relationship| core_index::RecordRelationship {
                predicate: relationship.predicate,
                record_id: relationship.record_id,
                collection: relationship.collection,
            })
            .collect(),
    }
}

/// Converts a core record to an FFI record.
fn ffi_record(record: core_index::Record) -> Record {
    Record {
        id: record.id,
        collection: record.collection,
        title: record.title,
        content: record.content,
        metadata: record
            .metadata
            .into_iter()
            .map(|(key, value)| (key, value.into()))
            .collect(),
        relationships: record
            .relationships
            .into_iter()
            .map(|relationship| RecordRelationship {
                predicate: relationship.predicate,
                record_id: relationship.record_id,
                collection: relationship.collection,
            })
            .collect(),
    }
}

/// Converts an FFI search query to a core query.
fn core_search_query(
    query: IndexSearchQuery,
) -> Result<core_index::IndexSearchQuery, IndexError> {
    Ok(core_index::IndexSearchQuery {
        text: query.text,
        collections: query.collections,
        filter: query.filter.map(core_metadata_filter),
        limit: usize::try_from(query.limit).map_err(to_error)?,
    })
}

/// Converts a core search result to an FFI search result.
fn ffi_search_result(
    result: core_index::IndexSearchResult,
) -> IndexSearchResult {
    IndexSearchResult {
        record_id: result.record_id,
        collection: result.collection,
        title: result.title,
        metadata: result
            .metadata
            .into_iter()
            .map(|(key, value)| (key, value.into()))
            .collect(),
        score: result.score,
        preview: result.preview,
    }
}

/// Converts an FFI metadata filter to a core filter.
fn core_metadata_filter(filter: MetadataFilter) -> core_index::MetadataFilter {
    match filter {
        MetadataFilter::Equals { key, value } => {
            core_index::MetadataFilter::Equals {
                key,
                value: value.into(),
            }
        }
        MetadataFilter::All { filters } => core_index::MetadataFilter::All(
            filters.into_iter().map(core_metadata_filter).collect(),
        ),
    }
}

impl From<MetadataValue> for core_index::MetadataValue {
    fn from(value: MetadataValue) -> Self {
        match value {
            MetadataValue::Text { value } => Self::Text(value),
            MetadataValue::Integer { value } => Self::Integer(value),
            MetadataValue::Float { value } => Self::Float(value),
            MetadataValue::Boolean { value } => Self::Boolean(value),
        }
    }
}

impl From<core_index::MetadataValue> for MetadataValue {
    fn from(value: core_index::MetadataValue) -> Self {
        match value {
            core_index::MetadataValue::Text(value) => Self::Text { value },
            core_index::MetadataValue::Integer(value) => {
                Self::Integer { value }
            }
            core_index::MetadataValue::Float(value) => Self::Float { value },
            core_index::MetadataValue::Boolean(value) => {
                Self::Boolean { value }
            }
        }
    }
}

impl From<core_index::IndexAuditOperation> for IndexAuditOperation {
    fn from(value: core_index::IndexAuditOperation) -> Self {
        match value {
            core_index::IndexAuditOperation::Add => Self::Add,
            core_index::IndexAuditOperation::Remove => Self::Remove,
            core_index::IndexAuditOperation::Search => Self::Search,
        }
    }
}

impl From<core_index::IndexAuditRecord> for IndexAuditRecord {
    fn from(value: core_index::IndexAuditRecord) -> Self {
        Self {
            operation: value.operation.into(),
            engine: value.engine,
            duration_ms: value.duration_ms,
            outputs: value.outputs.into_iter().collect(),
        }
    }
}

/// Converts an error into an FFI error.
fn to_error(error: impl ToString) -> IndexError {
    IndexError::Runtime {
        message: error.to_string(),
    }
}
