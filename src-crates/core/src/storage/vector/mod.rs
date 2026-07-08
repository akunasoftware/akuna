//! Vector storage domain types and traits.
//!
//! ```no_run
//! use akuna_core::storage::vector::{VectorContextOptions, in_memory_context};
//!
//! # async fn example() -> Result<(), akuna_core::storage::VectorError> {
//! let _ = in_memory_context(384, &VectorContextOptions::default()).await?;
//! # Ok(())
//! # }
//! ```

mod backend;
mod error;

#[cfg(test)]
mod tests;

use std::path::Path;

use async_trait::async_trait;

use crate::metadata::{Metadata, MetadataFilter};

pub use error::{
    VectorError, VectorTarget, VectorTextIndex, VectorWriteOperation,
};

/// Chunk row stored for dense retrieval.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkEntry {
    /// Stable chunk identifier within a record.
    pub chunk_id: String,
    /// Stable record identifier within a collection.
    pub record_id: String,
    /// Collection containing the record.
    pub collection: String,
    /// Chunk order within the record.
    pub sequence: u32,
    /// Chunk text used for retrieval evidence.
    pub text: String,
    /// Dense chunk embedding.
    pub embedding: Vec<f32>,
    /// Record metadata copied onto the chunk.
    pub metadata: Metadata,
}

/// Record row stored for content hydration.
#[derive(Clone, Debug, PartialEq)]
pub struct RecordEntry {
    /// Stable record identifier within a collection.
    pub record_id: String,
    /// Collection containing the record.
    pub collection: String,
    /// Record title.
    pub title: String,
    /// Dense title embedding.
    pub title_embedding: Vec<f32>,
    /// Full record content.
    pub content: String,
    /// Record metadata.
    pub metadata: Metadata,
}

/// Dense vector search request.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorSearchQuery {
    /// Query embedding.
    pub embedding: Vec<f32>,
    /// Collections to search, or all collections when empty.
    pub collections: Vec<String>,
    /// Optional metadata predicate.
    pub filter: Option<MetadataFilter>,
    /// Maximum result count.
    pub limit: usize,
}

/// Full-text search request.
#[derive(Clone, Debug, PartialEq)]
pub struct TextSearchQuery {
    /// Query text.
    pub text: String,
    /// Collections to search, or all collections when empty.
    pub collections: Vec<String>,
    /// Optional metadata predicate.
    pub filter: Option<MetadataFilter>,
    /// Maximum result count.
    pub limit: usize,
}

/// Vector context configuration.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VectorContextOptions {
    /// Enables full-text search over chunks.
    pub chunk_text_index: bool,
    /// Enables full-text search over titles.
    pub title_text_index: bool,
}

/// Chunk search result.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkSearchResult {
    /// Matching chunk identifier.
    pub chunk_id: String,
    /// Matching record identifier.
    pub record_id: String,
    /// Matching collection.
    pub collection: String,
    /// Matching chunk text.
    pub text: String,
    /// Matching record metadata.
    pub metadata: Metadata,
    /// Search score.
    pub score: f32,
}

/// Title search result.
#[derive(Clone, Debug, PartialEq)]
pub struct RecordSearchResult {
    /// Matching record identifier.
    pub record_id: String,
    /// Matching collection.
    pub collection: String,
    /// Matching record title.
    pub title: String,
    /// Matching record metadata.
    pub metadata: Metadata,
    /// Search score.
    pub score: f32,
}

/// Vector storage context.
#[async_trait]
pub trait VectorDbContext: Send + Sync {
    /// Atomically replaces all vector rows for one record.
    async fn put_record_with_chunks(
        &self,
        record: &RecordEntry,
        chunks: &[ChunkEntry],
    ) -> Result<(), VectorError>;

    /// Replaces all chunk rows for one record.
    async fn put_chunks(
        &self,
        collection: &str,
        record_id: &str,
        chunks: &[ChunkEntry],
    ) -> Result<(), VectorError>;

    /// Stores or replaces the record row.
    async fn put_record(&self, record: &RecordEntry)
    -> Result<(), VectorError>;

    /// Deletes all rows for a record.
    async fn delete_record(
        &self,
        collection: &str,
        record_id: &str,
    ) -> Result<(), VectorError>;

    /// Reads one record row.
    async fn get_record(
        &self,
        collection: &str,
        record_id: &str,
    ) -> Result<Option<RecordEntry>, VectorError>;

    /// Reads many record rows.
    async fn get_records(
        &self,
        keys: &[(String, String)],
    ) -> Result<Vec<RecordEntry>, VectorError>;

    /// Searches chunk embeddings.
    async fn search_chunks(
        &self,
        query: &VectorSearchQuery,
    ) -> Result<Vec<ChunkSearchResult>, VectorError>;

    /// Searches title embeddings.
    async fn search_titles(
        &self,
        query: &VectorSearchQuery,
    ) -> Result<Vec<RecordSearchResult>, VectorError>;

    /// Searches chunk text.
    async fn search_chunks_text(
        &self,
        query: &TextSearchQuery,
    ) -> Result<Vec<ChunkSearchResult>, VectorError>;

    /// Searches title text.
    async fn search_titles_text(
        &self,
        query: &TextSearchQuery,
    ) -> Result<Vec<RecordSearchResult>, VectorError>;
}

/// Opens a persistent vector storage context rooted at `path`.
pub async fn open_context(
    path: impl AsRef<Path>,
    dimensions: usize,
    options: &VectorContextOptions,
) -> Result<Box<dyn VectorDbContext>, VectorError> {
    Ok(Box::new(
        backend::TursoDbContext::open(
            path.as_ref().to_path_buf(),
            dimensions,
            options,
        )
        .await?,
    ))
}

/// Creates an ephemeral vector storage context.
pub async fn in_memory_context(
    dimensions: usize,
    options: &VectorContextOptions,
) -> Result<Box<dyn VectorDbContext>, VectorError> {
    Ok(Box::new(
        backend::TursoDbContext::open_in_memory(dimensions, options).await?,
    ))
}
