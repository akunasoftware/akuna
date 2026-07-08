//! Embedded record index.
//!
//! ```no_run
//! use akuna_core::index::{Index, IndexOptions};
//!
//! # async fn example() -> Result<(), akuna_core::index::IndexError> {
//! let _ = Index::new(IndexOptions::default()).await?;
//! # Ok(())
//! # }
//! ```

mod error;
mod manifest;

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::OsStr;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

use crate::chunking::ChunkingOptions;
use crate::chunking::packer::pack;
use crate::chunking::prose::segment_prose;
use crate::embedding::{EmbeddingModel, TextEmbedder, TextEmbedderOptions};
use crate::index::manifest::StoredConfig;
use crate::reranking::{RerankingModel, TextReranker, TextRerankerOptions};
use crate::storage::graph::open_context as open_graph_context;
use crate::storage::vector::{
    TextSearchQuery, VectorContextOptions, open_context as open_vector_context,
};
use crate::storage::{
    ChunkEntry, ChunkSearchResult, GraphDbContext, GraphEdge, GraphError,
    GraphNode, RecordEntry, RecordSearchResult, VectorDbContext,
    VectorSearchQuery,
};

pub use crate::metadata::{Metadata, MetadataFilter, MetadataValue};
pub use error::IndexError;

type Result<T> = anyhow::Result<T>;

const RECORD_LABEL: &str = "Record";
const VECTOR_DIR: &str = "vector";
const GRAPH_DIR: &str = "graph";
const COLLECTION_LABEL_PREFIX: &str = "Collection_";
const RELATIONSHIP_TYPE_PREFIX: &str = "Relationship_";
const GRAPH_METADATA_KEY: &str = "metadata_json";
const CANDIDATE_MULTIPLIER: usize = 4;
const MIN_CANDIDATES: usize = 20;
const RRF_K: f32 = 60.0;
const DENSE_CHUNK_PRIORITY: usize = 0;
const TEXT_CHUNK_PRIORITY: usize = 1;
const EXPANSION_MULTIPLIER: usize = 2;
const EXPANDED_SCORE_DAMPING: f32 = 0.5;
const LEADING_WINDOW_CHARS: usize = 1500;
const PREVIEW_MAX_CHARS: usize = 240;
const ELLIPSIS: char = '…';

/// Options for opening an [`Index`].
#[derive(Clone, Debug, PartialEq)]
pub struct IndexOptions {
    /// Storage subpath under the data root.
    pub name: String,
    /// Data root for persistent storage.
    pub path: Option<PathBuf>,
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
    pub cache_dir: Option<PathBuf>,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            path: None,
            embedding_model: EmbeddingModel::default(),
            reranking_model: Some(RerankingModel::default()),
            fulltext: true,
            graph: true,
            chunking: ChunkingOptions::default(),
            cache_dir: None,
        }
    }
}

/// Record stored in an [`Index`].
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, utoipa::ToSchema)]
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
    pub metadata: Metadata,
    /// Outgoing record relationships.
    pub relationships: Vec<RecordRelationship>,
}

/// Relationship from one record to another.
#[derive(
    Clone, Debug, PartialEq, Eq, Deserialize, Serialize, utoipa::ToSchema,
)]
pub struct RecordRelationship {
    /// Relationship predicate.
    pub predicate: String,
    /// Target record id.
    pub record_id: String,
    /// Target collection.
    pub collection: String,
}

/// Search request for an [`Index`].
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, utoipa::ToSchema)]
pub struct IndexSearchQuery {
    /// Query text.
    pub text: String,
    /// Collections to search.
    pub collections: Vec<String>,
    /// Metadata predicate.
    pub filter: Option<MetadataFilter>,
    /// Maximum result count.
    pub limit: usize,
}

impl Default for IndexSearchQuery {
    fn default() -> Self {
        Self {
            text: String::new(),
            collections: Vec::new(),
            filter: None,
            limit: 10,
        }
    }
}

/// Search result from an [`Index`].
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, utoipa::ToSchema)]
pub struct IndexSearchResult {
    /// Matching record id.
    pub record_id: String,
    /// Matching collection.
    pub collection: String,
    /// Matching title.
    pub title: String,
    /// Matching metadata.
    pub metadata: Metadata,
    /// Relevance score.
    pub score: f32,
    /// Matching content preview.
    pub preview: Option<String>,
}

/// Index operation recorded for auditing.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, utoipa::ToSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum IndexAuditOperation {
    /// Records were added or replaced.
    Add,
    /// A record was removed.
    Remove,
    /// Records were searched.
    Search,
}

/// Metered index operation record.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, utoipa::ToSchema)]
pub struct IndexAuditRecord {
    /// Index operation that completed.
    pub operation: IndexAuditOperation,
    /// Engine that performed the operation.
    pub engine: String,
    /// Wall-clock operation duration in milliseconds.
    pub duration_ms: u64,
    /// Output counts keyed by output kind.
    pub outputs: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CandidateKey {
    collection: String,
    record_id: String,
}

#[derive(Clone, Debug, Default)]
struct CandidateAccumulator {
    fused_score: f32,
    title_hit: bool,
    chunks: Vec<String>,
    chunk_ids: HashSet<String>,
    best_chunk: Option<BestChunk>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BestChunk {
    rank: usize,
    priority: usize,
    text: String,
}

#[derive(Clone, Debug)]
struct FusedCandidate {
    key: CandidateKey,
    fused_score: f32,
    title_hit: bool,
    chunks: Vec<String>,
    best_chunk: Option<BestChunk>,
}

#[derive(Clone, Debug)]
struct RankingCandidate {
    record_id: String,
    collection: String,
    title: String,
    content: String,
    metadata: Metadata,
    fused_score: f32,
    title_hit: bool,
    chunks: Vec<String>,
    best_chunk: Option<BestChunk>,
}

#[derive(Clone, Debug, PartialEq)]
enum CandidateEvidence {
    Chunk(String),
    Title,
    LeadingWindow(String),
    None,
}

#[derive(Clone, Debug)]
struct SearchCandidate {
    record_id: String,
    collection: String,
    title: String,
    content: String,
    metadata: Metadata,
    score: f32,
    best_evidence: CandidateEvidence,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PreviewToken {
    term: String,
    start: usize,
    end: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PreviewWindow {
    start: usize,
    end: usize,
    distinct: usize,
    occurrences: usize,
}

#[derive(Clone, Debug)]
struct ExpansionSeed {
    key: CandidateKey,
    score: f32,
}

#[derive(Clone, Debug)]
struct GraphExpansionNeighbor {
    key: CandidateKey,
    metadata: Metadata,
}

#[derive(Clone, Debug)]
struct ExpansionCandidate {
    key: CandidateKey,
    parent_score: f32,
}

#[derive(Clone, Debug)]
struct PreparedRecord {
    record: Record,
    node: Option<GraphNode>,
    vector: RecordEntry,
    chunks: Vec<ChunkEntry>,
}

#[derive(Clone, Debug, Default)]
struct IndexWriteCounts {
    chunks: usize,
    records: usize,
    relationships: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct EvidenceRef {
    candidate_index: usize,
    evidence: CandidateEvidence,
}

/// Embedded retrieval index.
pub struct Index {
    vector: Box<dyn VectorDbContext>,
    graph: Option<Box<dyn GraphDbContext>>,
    embedder: TextEmbedder,
    reranker: Option<Arc<TextReranker>>,
    fulltext: bool,
    chunking: ChunkingOptions,
    audit: Mutex<Vec<IndexAuditRecord>>,
    _temp_dir: Option<TempDir>,
}

impl Index {
    /// Opens an index from options.
    pub async fn new(
        options: IndexOptions,
    ) -> std::result::Result<Self, IndexError> {
        validate_name(&options.name)?;
        Self::new_inner(options).await.map_err(IndexError::open)
    }

    /// Opens an index after validating its public options.
    async fn new_inner(options: IndexOptions) -> Result<Self> {
        let (data_root, temp_dir) = data_root(options.path.as_deref())?;
        let root = data_root.join(&options.name);
        let config = StoredConfig::from_options(&options);
        manifest::ensure(&root, &config)?;

        let embedder = TextEmbedder::new(TextEmbedderOptions {
            model: options.embedding_model,
            cache_dir: options.cache_dir.clone(),
        })
        .await
        .context("failed to load index embedder")?;
        let reranker = if let Some(model) = options.reranking_model {
            Some(Arc::new(
                TextReranker::new(TextRerankerOptions {
                    model,
                    cache_dir: options.cache_dir.clone(),
                })
                .await
                .context("failed to load index reranker")?,
            ))
        } else {
            None
        };
        let dimensions = embedder
            .embed("dimension probe")
            .context("failed to determine embedding dimensions")?
            .len();

        let vector_path = root.join(VECTOR_DIR);
        let vector = open_vector_context(
            &vector_path,
            dimensions,
            &VectorContextOptions {
                chunk_text_index: options.fulltext,
                title_text_index: options.fulltext,
            },
        )
        .await
        .context("failed to open index vector storage")?;

        let graph = if options.graph {
            let graph_path = root.join(GRAPH_DIR);
            fs::create_dir_all(&graph_path).with_context(|| {
                format!("failed to create graph root {}", graph_path.display())
            })?;
            Some(
                open_graph_context(&graph_path)
                    .context("failed to open index graph storage")?,
            )
        } else {
            None
        };

        Ok(Self {
            vector,
            graph,
            embedder,
            reranker,
            fulltext: options.fulltext,
            chunking: options.chunking,
            audit: Mutex::new(Vec::new()),
            _temp_dir: temp_dir,
        })
    }

    /// Builds an index with injected dependencies for deterministic tests.
    #[cfg(test)]
    pub(crate) fn with_dependencies(
        vector: Box<dyn VectorDbContext>,
        graph: Option<Box<dyn GraphDbContext>>,
        embedder: TextEmbedder,
        reranker: Option<Arc<TextReranker>>,
        fulltext: bool,
        chunking: ChunkingOptions,
    ) -> Self {
        Self {
            vector,
            graph,
            embedder,
            reranker,
            fulltext,
            chunking,
            audit: Mutex::new(Vec::new()),
            _temp_dir: None,
        }
    }

    /// Adds or replaces records.
    pub async fn add(
        &self,
        records: Vec<Record>,
    ) -> std::result::Result<(), IndexError> {
        let started = Instant::now();
        self.validate_records(&records)?;
        self.preflight_relationship_targets(&records).await?;
        let counts =
            self.add_inner(records).await.map_err(IndexError::write)?;
        self.record_audit(
            IndexAuditOperation::Add,
            started,
            BTreeMap::from([
                ("chunks".to_string(), counts.chunks as u64),
                ("records".to_string(), counts.records as u64),
                ("relationships".to_string(), counts.relationships as u64),
            ]),
        );

        Ok(())
    }

    /// Removes one record.
    pub async fn remove(
        &self,
        collection: &str,
        record_id: &str,
    ) -> std::result::Result<(), IndexError> {
        let started = Instant::now();
        let outputs = self
            .remove_inner(collection, record_id)
            .await
            .map_err(IndexError::write)?;
        self.record_audit(IndexAuditOperation::Remove, started, outputs);

        Ok(())
    }

    /// Reads one record.
    pub async fn get(
        &self,
        collection: &str,
        record_id: &str,
    ) -> std::result::Result<Option<Record>, IndexError> {
        self.get_inner(collection, record_id)
            .await
            .map_err(IndexError::read)
    }

    /// Reads one record without converting the internal error.
    async fn get_inner(
        &self,
        collection: &str,
        record_id: &str,
    ) -> Result<Option<Record>> {
        let Some(entry) = self
            .vector
            .get_record(collection, record_id)
            .await
            .with_context(|| {
                format!(
                    "failed to read vector record '{record_id}' in collection '{collection}'",
                )
            })?
        else {
            return Ok(None);
        };

        let relationships = if let Some(graph) = self.graph.as_ref() {
            relationships(graph.as_ref(), collection, record_id)?
        } else {
            Vec::new()
        };

        Ok(Some(Record {
            id: entry.record_id,
            collection: entry.collection,
            title: entry.title,
            content: entry.content,
            metadata: entry.metadata,
            relationships,
        }))
    }

    /// Searches records.
    pub async fn search(
        &self,
        query: IndexSearchQuery,
    ) -> std::result::Result<Vec<IndexSearchResult>, IndexError> {
        if query.text.trim().is_empty() {
            return Err(IndexError::InvalidInput {
                message: "search text must not be empty".to_string(),
            });
        }
        let started = Instant::now();
        let results =
            self.search_inner(query).await.map_err(IndexError::search)?;
        self.record_audit(
            IndexAuditOperation::Search,
            started,
            BTreeMap::from([("results".to_string(), results.len() as u64)]),
        );

        Ok(results)
    }

    /// Searches records without converting the internal error.
    async fn search_inner(
        &self,
        query: IndexSearchQuery,
    ) -> Result<Vec<IndexSearchResult>> {
        let text = query.text.trim();
        if query.limit == 0 {
            return Ok(Vec::new());
        }

        let budget = query
            .limit
            .saturating_mul(CANDIDATE_MULTIPLIER)
            .max(MIN_CANDIDATES);
        let embedding = self
            .embedder
            .embed(text)
            .context("failed to embed search query")?;
        let candidates = self
            .retrieve_candidates(text, &query, embedding, budget)
            .await?;
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let candidates = self.hydrate_candidates(candidates).await?;
        let candidates =
            self.rank_candidates(text.to_string(), candidates).await?;
        let mut candidates = expand_candidates(
            self.graph.as_deref(),
            self.vector.as_ref(),
            self.reranker.clone(),
            text,
            &query,
            candidates,
        )
        .await?;
        candidates.sort_by(compare_candidates);
        candidates.truncate(query.limit);

        Ok(candidates
            .into_iter()
            .map(|candidate| IndexSearchResult {
                preview: candidate_preview(text, &candidate),
                record_id: candidate.record_id,
                collection: candidate.collection,
                title: candidate.title,
                metadata: candidate.metadata,
                score: candidate.score,
            })
            .collect())
    }

    /// Retrieves and fuses raw search candidates.
    async fn retrieve_candidates(
        &self,
        text: &str,
        query: &IndexSearchQuery,
        embedding: Vec<f32>,
        budget: usize,
    ) -> Result<Vec<FusedCandidate>> {
        let vector_query = VectorSearchQuery {
            embedding,
            collections: query.collections.clone(),
            filter: query.filter.clone(),
            limit: budget,
        };
        let text_query = TextSearchQuery {
            text: text.to_string(),
            collections: query.collections.clone(),
            filter: query.filter.clone(),
            limit: budget,
        };
        let mut candidates = HashMap::new();

        add_chunk_results(
            &mut candidates,
            self.vector
                .search_chunks(&vector_query)
                .await
                .context("failed to search chunk embeddings")?,
            DENSE_CHUNK_PRIORITY,
        );
        add_title_results(
            &mut candidates,
            self.vector
                .search_titles(&vector_query)
                .await
                .context("failed to search title embeddings")?,
        );
        if self.fulltext {
            add_chunk_results(
                &mut candidates,
                self.vector
                    .search_chunks_text(&text_query)
                    .await
                    .context("failed to search chunk text")?,
                TEXT_CHUNK_PRIORITY,
            );
            add_title_results(
                &mut candidates,
                self.vector
                    .search_titles_text(&text_query)
                    .await
                    .context("failed to search title text")?,
            );
        }

        Ok(candidates
            .into_iter()
            .map(|(key, candidate)| FusedCandidate {
                key,
                fused_score: candidate.fused_score,
                title_hit: candidate.title_hit,
                chunks: candidate.chunks,
                best_chunk: candidate.best_chunk,
            })
            .collect())
    }

    /// Hydrates fused candidates from record storage.
    async fn hydrate_candidates(
        &self,
        candidates: Vec<FusedCandidate>,
    ) -> Result<Vec<RankingCandidate>> {
        let keys = candidates
            .iter()
            .map(|candidate| {
                (
                    candidate.key.collection.clone(),
                    candidate.key.record_id.clone(),
                )
            })
            .collect::<Vec<_>>();
        let records = self
            .vector
            .get_records(&keys)
            .await
            .context("failed to hydrate search candidates")?
            .into_iter()
            .map(|record| {
                (
                    (record.collection.clone(), record.record_id.clone()),
                    record,
                )
            })
            .collect::<HashMap<_, _>>();

        Ok(candidates
            .into_iter()
            .filter_map(|candidate| {
                records
                    .get(&(
                        candidate.key.collection.clone(),
                        candidate.key.record_id.clone(),
                    ))
                    .map(|record| RankingCandidate {
                        record_id: record.record_id.clone(),
                        collection: record.collection.clone(),
                        title: record.title.clone(),
                        content: record.content.clone(),
                        metadata: record.metadata.clone(),
                        fused_score: candidate.fused_score,
                        title_hit: candidate.title_hit,
                        chunks: candidate.chunks,
                        best_chunk: candidate.best_chunk,
                    })
            })
            .collect())
    }

    /// Ranks hydrated candidates.
    async fn rank_candidates(
        &self,
        text: String,
        candidates: Vec<RankingCandidate>,
    ) -> Result<Vec<SearchCandidate>> {
        let Some(reranker) = self.reranker.clone() else {
            return Ok(candidates
                .into_iter()
                .map(candidate_without_rerank)
                .collect());
        };

        let (pairs, evidence_refs) = rerank_pairs(&text, &candidates);
        if pairs.is_empty() {
            return Ok(candidates
                .into_iter()
                .map(candidate_without_rerank)
                .collect());
        }

        let scores = tokio::task::spawn_blocking(move || {
            reranker.score_batch(&pairs, None)
        })
        .await
        .context("failed to join search reranker task")?
        .context("failed to rerank search evidence")?;

        apply_rerank_scores(candidates, evidence_refs, scores)
    }

    /// Returns completed operation audit records.
    pub fn audit_records(&self) -> Vec<IndexAuditRecord> {
        match self.audit.lock() {
            Ok(records) => records.clone(),
            Err(error) => error.into_inner().clone(),
        }
    }

    /// Records a completed index operation.
    fn record_audit(
        &self,
        operation: IndexAuditOperation,
        started: Instant,
        outputs: BTreeMap<String, u64>,
    ) {
        let record = IndexAuditRecord {
            operation,
            engine: "index".to_string(),
            duration_ms: started.elapsed().as_millis() as u64,
            outputs,
        };
        match self.audit.lock() {
            Ok(mut records) => records.push(record),
            Err(error) => error.into_inner().push(record),
        }
    }

    /// Validates a record batch before writes.
    fn validate_records(
        &self,
        records: &[Record],
    ) -> std::result::Result<(), IndexError> {
        for record in records {
            if self.graph.is_none() && !record.relationships.is_empty() {
                return Err(IndexError::InvalidInput {
                    message: "relationships require graph storage".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Checks every relationship target before mutating the batch.
    async fn preflight_relationship_targets(
        &self,
        records: &[Record],
    ) -> std::result::Result<(), IndexError> {
        let Some(graph) = self.graph.as_ref() else {
            return Ok(());
        };
        let batch = records
            .iter()
            .map(|record| (record.collection.as_str(), record.id.as_str()))
            .collect::<HashSet<_>>();

        for record in records {
            for relationship in &record.relationships {
                if batch.contains(&(
                    relationship.collection.as_str(),
                    relationship.record_id.as_str(),
                )) {
                    continue;
                }
                let labels = record_labels(&relationship.collection);
                let label_refs = label_refs(&labels);
                let target = graph
                    .get_node(&label_refs, &relationship.record_id)
                    .map_err(|error| IndexError::Write {
                        source: Box::new(error),
                    })?;
                if target.is_some() {
                    continue;
                }

                return Err(IndexError::InvalidInput {
                    message: format!(
                        "relationship target not found for record '{}' in collection '{}': target '{}' in collection '{}'",
                        record.id,
                        record.collection,
                        relationship.record_id,
                        relationship.collection,
                    ),
                });
            }
        }

        Ok(())
    }

    /// Adds a validated batch and restores preimages when any write fails.
    async fn add_inner(
        &self,
        records: Vec<Record>,
    ) -> Result<IndexWriteCounts> {
        let snapshots = self.record_snapshots(&records).await?;
        let prepared = records
            .iter()
            .map(|record| self.prepare_record(record))
            .collect::<Result<Vec<_>>>()?;
        let counts = IndexWriteCounts {
            chunks: prepared.iter().map(|record| record.chunks.len()).sum(),
            records: prepared.len(),
            relationships: prepared
                .iter()
                .map(|record| record.record.relationships.len())
                .sum(),
        };

        for record in &prepared {
            if let Err(error) = self.write_prepared_record(record).await {
                return Err(self.rollback_after_error(&snapshots, error).await);
            }
        }
        if let Some(graph) = self.graph.as_ref() {
            for record in &prepared {
                if let Err(error) =
                    self.replace_relationships(graph.as_ref(), &record.record)
                {
                    return Err(self
                        .rollback_after_error(&snapshots, error)
                        .await);
                }
            }
        }

        Ok(counts)
    }

    /// Removes one record and restores its vector and graph preimages on failure.
    async fn remove_inner(
        &self,
        collection: &str,
        record_id: &str,
    ) -> Result<BTreeMap<String, u64>> {
        let vector = self
            .get_inner(collection, record_id)
            .await?
            .map(|record| self.prepare_record(&record))
            .transpose()?;
        let (node, edges) = if let Some(graph) = self.graph.as_ref() {
            let labels = record_labels(collection);
            let label_refs = label_refs(&labels);
            let node = graph.get_node(&label_refs, record_id)?;
            let edges = graph
                .neighbors(&label_refs, record_id)?
                .into_iter()
                .map(|(edge, _)| edge)
                .collect::<Vec<_>>();
            (node, edges)
        } else {
            (None, Vec::new())
        };

        if let Err(error) =
            self.vector.delete_record(collection, record_id).await
        {
            return Err(self
                .rollback_remove(
                    vector.as_ref(),
                    node.as_ref(),
                    &edges,
                    error.into(),
                )
                .await);
        }
        if let Some(graph) = self.graph.as_ref() {
            let labels = record_labels(collection);
            let label_refs = label_refs(&labels);
            for edge in &edges {
                if let Err(error) =
                    delete_edge_absorb_missing(graph.as_ref(), edge)
                {
                    return Err(self
                        .rollback_remove(
                            vector.as_ref(),
                            node.as_ref(),
                            &edges,
                            error,
                        )
                        .await);
                }
            }
            if let Err(error) = delete_node_absorb_missing(
                graph.as_ref(),
                &label_refs,
                record_id,
            ) {
                return Err(self
                    .rollback_remove(
                        vector.as_ref(),
                        node.as_ref(),
                        &edges,
                        error,
                    )
                    .await);
            }
        }

        Ok(BTreeMap::from([
            ("edges".to_string(), edges.len() as u64),
            ("records".to_string(), u64::from(u8::from(vector.is_some()))),
        ]))
    }

    /// Builds vector and graph rows before any mutation can start.
    fn prepare_record(&self, record: &Record) -> Result<PreparedRecord> {
        let node = self
            .graph
            .is_some()
            .then(|| graph_node(record))
            .transpose()?;
        let (vector, chunks) = self.vector_entries(record)?;

        Ok(PreparedRecord {
            record: record.clone(),
            node,
            vector,
            chunks,
        })
    }

    /// Writes one prepared record node and atomically replaced vector rows.
    async fn write_prepared_record(
        &self,
        record: &PreparedRecord,
    ) -> Result<()> {
        self.vector
            .put_record_with_chunks(&record.vector, &record.chunks)
            .await
            .with_context(|| {
                format!(
                    "failed to write vector record '{}' in collection '{}'",
                    record.record.id, record.record.collection,
                )
            })?;
        if let (Some(graph), Some(node)) =
            (self.graph.as_ref(), record.node.as_ref())
        {
            graph.put_node(node).with_context(|| {
                format!(
                    "failed to write graph node for record '{}' in collection '{}'",
                    record.record.id, record.record.collection,
                )
            })?;
        }

        Ok(())
    }

    /// Captures preimages for a record batch once per record identity.
    async fn record_snapshots(
        &self,
        records: &[Record],
    ) -> Result<HashMap<CandidateKey, Option<PreparedRecord>>> {
        let mut snapshots = HashMap::new();
        for record in records {
            let key = CandidateKey {
                collection: record.collection.clone(),
                record_id: record.id.clone(),
            };
            if snapshots.contains_key(&key) {
                continue;
            }
            let snapshot = self
                .get_inner(&record.collection, &record.id)
                .await?
                .map(|record| self.prepare_record(&record))
                .transpose()?;
            snapshots.insert(key, snapshot);
        }

        Ok(snapshots)
    }

    /// Restores batch preimages and retains the original failure as the source.
    async fn rollback_after_error(
        &self,
        snapshots: &HashMap<CandidateKey, Option<PreparedRecord>>,
        error: anyhow::Error,
    ) -> anyhow::Error {
        match self.rollback_records(snapshots).await {
            Ok(()) => error,
            Err(rollback) => {
                error.context(format!("index rollback failed: {rollback}"))
            }
        }
    }

    /// Restores vector rows, graph nodes, and outgoing relationships from preimages.
    async fn rollback_records(
        &self,
        snapshots: &HashMap<CandidateKey, Option<PreparedRecord>>,
    ) -> Result<()> {
        let mut keys = snapshots.keys().cloned().collect::<Vec<_>>();
        keys.sort_by(compare_keys);
        for key in &keys {
            let snapshot = snapshots
                .get(key)
                .context("index rollback snapshot missing")?;
            if let Some(record) = snapshot {
                self.write_prepared_record(record).await?;
                continue;
            }

            self.vector
                .delete_record(&key.collection, &key.record_id)
                .await?;
            if let Some(graph) = self.graph.as_ref() {
                let labels = record_labels(&key.collection);
                let label_refs = label_refs(&labels);
                delete_touching_edges(
                    graph.as_ref(),
                    &label_refs,
                    &key.record_id,
                )?;
                delete_node_absorb_missing(
                    graph.as_ref(),
                    &label_refs,
                    &key.record_id,
                )?;
            }
        }
        if let Some(graph) = self.graph.as_ref() {
            for key in &keys {
                let Some(record) = snapshots
                    .get(key)
                    .context("index rollback snapshot missing")?
                    .as_ref()
                else {
                    continue;
                };
                self.replace_relationships(graph.as_ref(), &record.record)?;
            }
        }

        Ok(())
    }

    /// Restores a failed removal and retains the original failure as the source.
    async fn rollback_remove(
        &self,
        vector: Option<&PreparedRecord>,
        node: Option<&GraphNode>,
        edges: &[GraphEdge],
        error: anyhow::Error,
    ) -> anyhow::Error {
        let rollback = async {
            if let Some(vector) = vector {
                self.vector
                    .put_record_with_chunks(&vector.vector, &vector.chunks)
                    .await?;
            }
            if let Some(graph) = self.graph.as_ref() {
                if let Some(node) = node {
                    graph.put_node(node)?;
                }
                for edge in edges {
                    graph.put_edge(edge)?;
                }
            }

            Ok::<_, anyhow::Error>(())
        }
        .await;

        match rollback {
            Ok(()) => error,
            Err(rollback) => {
                error.context(format!("index rollback failed: {rollback}"))
            }
        }
    }

    /// Builds vector rows for one record.
    fn vector_entries(
        &self,
        record: &Record,
    ) -> Result<(RecordEntry, Vec<ChunkEntry>)> {
        let chunks = pack(
            &record.content,
            &segment_prose(&record.content),
            &self.chunking,
        );
        let mut documents = Vec::with_capacity(chunks.len() + 1);
        documents.push(record.title.as_str());
        documents.extend(chunks.iter().map(|chunk| chunk.text.as_str()));
        let embeddings = self
            .embedder
            .embed_batch(&documents, None)
            .with_context(|| {
                format!(
                    "failed to embed record '{}' in collection '{}'",
                    record.id, record.collection,
                )
            })?;
        if embeddings.len() != documents.len() {
            bail!(
                "embedding count mismatch for record '{}' in collection '{}'",
                record.id,
                record.collection,
            );
        }
        let mut embeddings = embeddings.into_iter();
        let title_embedding = embeddings
            .next()
            .context("missing title embedding for record")?;
        let chunk_entries = chunks
            .into_iter()
            .zip(embeddings)
            .enumerate()
            .map(|(sequence, (chunk, embedding))| {
                let sequence = u32::try_from(sequence)
                    .context("record produced too many chunks")?;
                Ok(ChunkEntry {
                    chunk_id: format!("{}:{sequence}", record.id),
                    record_id: record.id.clone(),
                    collection: record.collection.clone(),
                    sequence,
                    text: chunk.text,
                    embedding,
                    metadata: record.metadata.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok((
            RecordEntry {
                record_id: record.id.clone(),
                collection: record.collection.clone(),
                title: record.title.clone(),
                title_embedding,
                content: record.content.clone(),
                metadata: record.metadata.clone(),
            },
            chunk_entries,
        ))
    }

    /// Replaces outgoing relationships for one record.
    fn replace_relationships(
        &self,
        graph: &dyn GraphDbContext,
        record: &Record,
    ) -> Result<()> {
        validate_relationship_targets(graph, record)?;
        let source_labels = record_labels(&record.collection);
        let source_label_refs = label_refs(&source_labels);
        for (edge, _node) in graph
            .neighbors(&source_label_refs, &record.id)
            .with_context(|| {
                format!(
                    "failed to read relationships for record '{}' in collection '{}'",
                    record.id, record.collection,
                )
            })?
        {
            if edge.source == record.id
                && labels_match(&edge.source_labels, &source_labels)
            {
                delete_edge_absorb_missing(graph, &edge)?;
            }
        }

        for relationship in &record.relationships {
            let target_labels = record_labels(&relationship.collection);
            graph
                .put_edge(&GraphEdge {
                    source_labels: source_labels.clone(),
                    source: record.id.clone(),
                    predicate: graph_token(
                        RELATIONSHIP_TYPE_PREFIX,
                        &relationship.predicate,
                    ),
                    target: relationship.record_id.clone(),
                    target_labels,
                })
                .with_context(|| {
                    format!(
                        "failed to write relationship for record '{}' in collection '{}'",
                        record.id, record.collection,
                    )
                })?;
        }

        Ok(())
    }
}

/// Validates every outgoing relationship target before replacing edges.
fn validate_relationship_targets(
    graph: &dyn GraphDbContext,
    record: &Record,
) -> Result<()> {
    for relationship in &record.relationships {
        let target_labels = record_labels(&relationship.collection);
        let target_label_refs = label_refs(&target_labels);
        if graph
            .get_node(&target_label_refs, &relationship.record_id)
            .with_context(|| {
                format!(
                    "failed to read relationship target for record '{}' in collection '{}'",
                    record.id, record.collection,
                )
            })?
            .is_some()
        {
            continue;
        }

        bail!(
            "relationship target not found for record '{}' in collection '{}': target '{}' in collection '{}'",
            record.id,
            record.collection,
            relationship.record_id,
            relationship.collection,
        );
    }

    Ok(())
}

/// Adds a chunk retrieval list to fusion state.
fn add_chunk_results(
    candidates: &mut HashMap<CandidateKey, CandidateAccumulator>,
    results: Vec<ChunkSearchResult>,
    priority: usize,
) {
    let mut seen_records = HashSet::new();
    let mut rank = 0usize;
    for result in results {
        let key = CandidateKey {
            collection: result.collection,
            record_id: result.record_id,
        };
        let candidate = candidates.entry(key.clone()).or_default();
        if candidate.chunk_ids.insert(result.chunk_id) {
            candidate.chunks.push(result.text.clone());
        }
        if !seen_records.insert(key) {
            continue;
        }

        rank += 1;
        candidate.fused_score += rrf_score(rank);
        let replace = candidate.best_chunk.as_ref().is_none_or(|best| {
            rank < best.rank || (rank == best.rank && priority < best.priority)
        });
        if replace {
            candidate.best_chunk = Some(BestChunk {
                rank,
                priority,
                text: result.text,
            });
        }
    }
}

/// Adds a title retrieval list to fusion state.
fn add_title_results(
    candidates: &mut HashMap<CandidateKey, CandidateAccumulator>,
    results: Vec<RecordSearchResult>,
) {
    let mut seen_records = HashSet::new();
    let mut rank = 0usize;
    for result in results {
        let key = CandidateKey {
            collection: result.collection,
            record_id: result.record_id,
        };
        if !seen_records.insert(key.clone()) {
            continue;
        }

        rank += 1;
        let candidate = candidates.entry(key).or_default();
        candidate.fused_score += rrf_score(rank);
        candidate.title_hit = true;
    }
}

/// Scores a reciprocal rank contribution.
fn rrf_score(rank: usize) -> f32 {
    1.0 / (RRF_K + rank as f32)
}

/// Builds reranker pair inputs.
fn rerank_pairs(
    text: &str,
    candidates: &[RankingCandidate],
) -> (Vec<(String, String)>, Vec<EvidenceRef>) {
    let mut pairs = Vec::new();
    let mut evidence_refs = Vec::new();
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        for chunk in &candidate.chunks {
            pairs.push((text.to_string(), chunk.clone()));
            evidence_refs.push(EvidenceRef {
                candidate_index,
                evidence: CandidateEvidence::Chunk(chunk.clone()),
            });
        }
        if candidate.title_hit {
            pairs.push((text.to_string(), candidate.title.clone()));
            evidence_refs.push(EvidenceRef {
                candidate_index,
                evidence: CandidateEvidence::Title,
            });
        }
    }

    (pairs, evidence_refs)
}

/// Applies reranker scores to hydrated candidates.
fn apply_rerank_scores(
    candidates: Vec<RankingCandidate>,
    evidence_refs: Vec<EvidenceRef>,
    scores: Vec<f32>,
) -> Result<Vec<SearchCandidate>> {
    if evidence_refs.len() != scores.len() {
        bail!("reranker score count mismatch");
    }

    let mut best = vec![None::<(f32, CandidateEvidence)>; candidates.len()];
    for (evidence_ref, score) in evidence_refs.into_iter().zip(scores) {
        let score = crate::ml::sigmoid_f32(score);
        let slot = best
            .get_mut(evidence_ref.candidate_index)
            .context("reranker evidence index out of range")?;
        if slot
            .as_ref()
            .is_none_or(|(best_score, _evidence)| score > *best_score)
        {
            *slot = Some((score, evidence_ref.evidence));
        }
    }

    Ok(candidates
        .into_iter()
        .zip(best)
        .map(|(candidate, best)| {
            let (score, best_evidence) = best
                .unwrap_or((candidate.fused_score, CandidateEvidence::None));
            SearchCandidate {
                record_id: candidate.record_id,
                collection: candidate.collection,
                title: candidate.title,
                content: candidate.content,
                metadata: candidate.metadata,
                score,
                best_evidence,
            }
        })
        .collect())
}

/// Builds a scored candidate without reranking.
fn candidate_without_rerank(candidate: RankingCandidate) -> SearchCandidate {
    let best_evidence = candidate.best_chunk.map_or_else(
        || {
            if candidate.title_hit {
                CandidateEvidence::Title
            } else {
                CandidateEvidence::None
            }
        },
        |chunk| CandidateEvidence::Chunk(chunk.text),
    );

    SearchCandidate {
        record_id: candidate.record_id,
        collection: candidate.collection,
        title: candidate.title,
        content: candidate.content,
        metadata: candidate.metadata,
        score: candidate.fused_score,
        best_evidence,
    }
}

/// Expands search candidates through the record graph.
async fn expand_candidates(
    graph: Option<&dyn GraphDbContext>,
    vector: &dyn VectorDbContext,
    reranker: Option<Arc<TextReranker>>,
    text: &str,
    query: &IndexSearchQuery,
    mut candidates: Vec<SearchCandidate>,
) -> Result<Vec<SearchCandidate>> {
    let Some(graph) = graph else {
        return Ok(candidates);
    };

    let expansions = collect_expansions(graph, query, &candidates)?;
    if expansions.is_empty() {
        // No graph additions leaves initial retrieval scores byte-for-byte inert.
        return Ok(candidates);
    }

    let mut expanded = hydrate_expansions(vector, expansions).await?;
    if expanded.is_empty() {
        return Ok(candidates);
    }

    candidates.append(&mut expanded);
    aggregate_candidates(reranker, text, candidates).await
}

/// Collects graph expansion keys.
fn collect_expansions(
    graph: &dyn GraphDbContext,
    query: &IndexSearchQuery,
    candidates: &[SearchCandidate],
) -> Result<Vec<ExpansionCandidate>> {
    let mut seen = candidates
        .iter()
        .map(search_candidate_key)
        .collect::<HashSet<_>>();
    let mut expansions = Vec::new();
    let max_expansions = query.limit.saturating_mul(EXPANSION_MULTIPLIER);

    for seed in expansion_seeds(candidates, query.limit) {
        if expansions.len() == max_expansions {
            break;
        }

        let labels = record_labels(&seed.key.collection);
        let label_refs = label_refs(&labels);
        let mut neighbors = graph
            .neighbors(&label_refs, &seed.key.record_id)
            .with_context(|| {
                format!(
                    "failed to expand graph record '{}' in collection '{}'",
                    seed.key.record_id, seed.key.collection,
                )
            })?
            .into_iter()
            .filter_map(|(_edge, node)| {
                graph_expansion_neighbor(node).transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        neighbors.sort_by(|left, right| compare_keys(&left.key, &right.key));

        for neighbor in neighbors {
            if expansions.len() == max_expansions {
                break;
            }
            if !query_matches_expansion(query, &neighbor) {
                continue;
            }
            if !seen.insert(neighbor.key.clone()) {
                continue;
            }

            expansions.push(ExpansionCandidate {
                key: neighbor.key,
                parent_score: seed.score,
            });
        }
    }

    Ok(expansions)
}

/// Returns expansion seeds in scoring order.
fn expansion_seeds(
    candidates: &[SearchCandidate],
    limit: usize,
) -> Vec<ExpansionSeed> {
    let mut seeds = candidates
        .iter()
        .map(|candidate| ExpansionSeed {
            key: search_candidate_key(candidate),
            score: candidate.score,
        })
        .collect::<Vec<_>>();
    seeds.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| compare_keys(&left.key, &right.key))
    });
    seeds.truncate(limit);

    seeds
}

/// Builds a candidate key.
fn search_candidate_key(candidate: &SearchCandidate) -> CandidateKey {
    CandidateKey {
        collection: candidate.collection.clone(),
        record_id: candidate.record_id.clone(),
    }
}

/// Builds an expansion neighbor from a graph node.
fn graph_expansion_neighbor(
    node: GraphNode,
) -> Result<Option<GraphExpansionNeighbor>> {
    if !node.labels.iter().any(|label| label == RECORD_LABEL) {
        return Ok(None);
    }
    let Some(collection) = collection_from_labels(&node.labels) else {
        return Ok(None);
    };

    let metadata = graph_metadata(&node.metadata)?;
    Ok(Some(GraphExpansionNeighbor {
        key: CandidateKey {
            collection,
            record_id: node.id,
        },
        metadata,
    }))
}

/// Converts graph payload into record metadata.
fn graph_metadata(value: &Option<serde_json::Value>) -> Result<Metadata> {
    let Some(value) = value else {
        return Ok(Metadata::new());
    };
    let serde_json::Value::Object(values) = value else {
        bail!("graph record metadata must be an object");
    };
    let Some(value) = values.get(GRAPH_METADATA_KEY) else {
        return Ok(Metadata::new());
    };
    let Some(value) = value.as_str() else {
        bail!("graph record metadata must be encoded as a string");
    };

    serde_json::from_str(value).context("failed to deserialize graph metadata")
}

/// Checks query scope for one expansion.
fn query_matches_expansion(
    query: &IndexSearchQuery,
    neighbor: &GraphExpansionNeighbor,
) -> bool {
    let collection_matches = query.collections.is_empty()
        || query
            .collections
            .iter()
            .any(|collection| collection == &neighbor.key.collection);
    collection_matches
        && query
            .filter
            .as_ref()
            .is_none_or(|filter| filter.matches(&neighbor.metadata))
}

/// Hydrates expanded records.
async fn hydrate_expansions(
    vector: &dyn VectorDbContext,
    expansions: Vec<ExpansionCandidate>,
) -> Result<Vec<SearchCandidate>> {
    let keys = expansions
        .iter()
        .map(|expansion| {
            (
                expansion.key.collection.clone(),
                expansion.key.record_id.clone(),
            )
        })
        .collect::<Vec<_>>();
    let records = vector
        .get_records(&keys)
        .await
        .context("failed to hydrate graph expansions")?
        .into_iter()
        .map(|record| {
            (
                (record.collection.clone(), record.record_id.clone()),
                record,
            )
        })
        .collect::<HashMap<_, _>>();

    Ok(expansions
        .into_iter()
        .filter_map(|expansion| {
            records
                .get(&(
                    expansion.key.collection.clone(),
                    expansion.key.record_id.clone(),
                ))
                .map(|record| {
                    expanded_candidate(record, expansion.parent_score)
                })
        })
        .collect())
}

/// Builds one hydrated expansion candidate.
fn expanded_candidate(
    record: &RecordEntry,
    parent_score: f32,
) -> SearchCandidate {
    let leading_window = leading_window(&record.content);
    SearchCandidate {
        record_id: record.record_id.clone(),
        collection: record.collection.clone(),
        title: record.title.clone(),
        content: record.content.clone(),
        metadata: record.metadata.clone(),
        score: parent_score * EXPANDED_SCORE_DAMPING,
        best_evidence: CandidateEvidence::LeadingWindow(leading_window),
    }
}

/// Returns a bounded leading content window.
fn leading_window(text: &str) -> String {
    text.chars().take(LEADING_WINDOW_CHARS).collect()
}

/// Applies final aggregate scoring.
async fn aggregate_candidates(
    reranker: Option<Arc<TextReranker>>,
    text: &str,
    candidates: Vec<SearchCandidate>,
) -> Result<Vec<SearchCandidate>> {
    let Some(reranker) = reranker else {
        return Ok(candidates);
    };

    let pairs = candidates
        .iter()
        .map(|candidate| (text.to_string(), representative_text(candidate)))
        .collect::<Vec<_>>();
    let scores =
        tokio::task::spawn_blocking(move || reranker.score_batch(&pairs, None))
            .await
            .context("failed to join final search reranker task")?
            .context("failed to rerank final search candidates")?;

    apply_final_scores(candidates, scores)
}

/// Applies final aggregate scores.
fn apply_final_scores(
    mut candidates: Vec<SearchCandidate>,
    scores: Vec<f32>,
) -> Result<Vec<SearchCandidate>> {
    if candidates.len() != scores.len() {
        bail!("final reranker score count mismatch");
    }

    for (candidate, score) in candidates.iter_mut().zip(scores) {
        candidate.score = crate::ml::sigmoid_f32(score);
    }

    Ok(candidates)
}

/// Builds final reranker text for one candidate.
fn representative_text(candidate: &SearchCandidate) -> String {
    match &candidate.best_evidence {
        CandidateEvidence::Chunk(text) => titled_text(&candidate.title, text),
        CandidateEvidence::Title | CandidateEvidence::None => {
            candidate.title.clone()
        }
        CandidateEvidence::LeadingWindow(text) => {
            titled_text(&candidate.title, text)
        }
    }
}

/// Joins a title and evidence text.
fn titled_text(title: &str, text: &str) -> String {
    if title.is_empty() {
        return text.to_string();
    }
    if text.is_empty() {
        return title.to_string();
    }

    format!("{title}\n{text}")
}

/// Builds a result preview for one candidate.
fn candidate_preview(
    query: &str,
    candidate: &SearchCandidate,
) -> Option<String> {
    match &candidate.best_evidence {
        CandidateEvidence::Chunk(text)
        | CandidateEvidence::LeadingWindow(text) => build_preview(query, text),
        CandidateEvidence::Title | CandidateEvidence::None => {
            build_preview(query, &leading_window(&candidate.content))
        }
    }
}

/// Builds a bounded preview from evidence text.
fn build_preview(query: &str, evidence: &str) -> Option<String> {
    let collapsed = collapse_whitespace(evidence);
    if collapsed.is_empty() {
        return None;
    }

    let total_chars = collapsed.chars().count();
    if total_chars <= PREVIEW_MAX_CHARS {
        return Some(collapsed);
    }

    let terms = query_terms(query);
    let tokens = preview_tokens(&collapsed);
    let Some(window) = best_preview_window(&tokens, &terms, total_chars) else {
        return Some(preview_head(&collapsed, total_chars));
    };
    if window.distinct == 0 {
        return Some(preview_head(&collapsed, total_chars));
    }
    let Some((focus_start, focus_end)) =
        focus_span(&tokens, &terms, window.start, window.end)
    else {
        return Some(preview_head(&collapsed, total_chars));
    };

    Some(preview_centered(
        &collapsed,
        total_chars,
        focus_start,
        focus_end,
    ))
}

/// Collapses whitespace runs for preview length math.
fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Returns deduplicated query terms.
fn query_terms(query: &str) -> HashSet<String> {
    preview_tokens(query)
        .into_iter()
        .map(|token| token.term)
        .collect()
}

/// Tokenizes text for preview scoring.
fn preview_tokens(text: &str) -> Vec<PreviewToken> {
    let mut tokens = Vec::new();
    let mut term = String::new();
    let mut start = None;
    let mut char_count = 0;

    for (index, character) in text.chars().enumerate() {
        char_count = index + 1;
        if character.is_alphanumeric() {
            if start.is_none() {
                start = Some(index);
            }
            term.extend(character.to_lowercase());
            continue;
        }

        if let Some(token_start) = start.take() {
            tokens.push(PreviewToken {
                term: std::mem::take(&mut term),
                start: token_start,
                end: index,
            });
        }
    }

    if let Some(token_start) = start {
        tokens.push(PreviewToken {
            term,
            start: token_start,
            end: char_count,
        });
    }

    tokens
}

/// Finds the highest-scoring preview window.
fn best_preview_window(
    tokens: &[PreviewToken],
    terms: &HashSet<String>,
    total_chars: usize,
) -> Option<PreviewWindow> {
    let mut best = None;
    for (index, token) in tokens.iter().enumerate() {
        let end = preview_window_end(total_chars, token.start);
        let mut present = HashSet::new();
        let mut occurrences = 0;

        for window_token in &tokens[index..] {
            if window_token.end > end {
                break;
            }
            if terms.contains(&window_token.term) {
                occurrences += 1;
                present.insert(window_token.term.as_str());
            }
        }

        let window = PreviewWindow {
            start: token.start,
            end,
            distinct: present.len(),
            occurrences,
        };
        if best.as_ref().is_none_or(|best: &PreviewWindow| {
            window.distinct > best.distinct
                || (window.distinct == best.distinct
                    && window.occurrences > best.occurrences)
        }) {
            best = Some(window);
        }
    }

    best
}

/// Returns the end char index for a scored preview window.
fn preview_window_end(total_chars: usize, start: usize) -> usize {
    let prefix_chars = usize::from(start > 0);
    let budget_without_suffix = PREVIEW_MAX_CHARS.saturating_sub(prefix_chars);
    let suffix_chars =
        usize::from(start.saturating_add(budget_without_suffix) < total_chars);
    let budget = PREVIEW_MAX_CHARS
        .saturating_sub(prefix_chars.saturating_add(suffix_chars));

    start.saturating_add(budget).min(total_chars)
}

/// Returns the matching token span inside a scored window.
fn focus_span(
    tokens: &[PreviewToken],
    terms: &HashSet<String>,
    start: usize,
    end: usize,
) -> Option<(usize, usize)> {
    let mut span = None;
    for token in tokens {
        if token.start < start
            || token.end > end
            || !terms.contains(&token.term)
        {
            continue;
        }

        span = Some(match span {
            Some((first, _last)) => (first, token.end),
            None => (token.start, token.end),
        });
    }

    span
}

/// Builds a head preview.
fn preview_head(text: &str, total_chars: usize) -> String {
    let end = PREVIEW_MAX_CHARS.saturating_sub(1).min(total_chars);
    cut_preview(text, total_chars, 0, end)
}

/// Builds a centered preview.
fn preview_centered(
    text: &str,
    total_chars: usize,
    focus_start: usize,
    focus_end: usize,
) -> String {
    let center = focus_start + focus_end.saturating_sub(focus_start) / 2;
    let (start, end) = centered_preview_range(total_chars, center);

    cut_preview(text, total_chars, start, end)
}

/// Returns a centered preview range.
fn centered_preview_range(total_chars: usize, center: usize) -> (usize, usize) {
    let mut budget = PREVIEW_MAX_CHARS.saturating_sub(2);
    let mut start = center.saturating_sub(budget / 2);
    let mut end = start.saturating_add(budget).min(total_chars);
    if end == total_chars {
        start = total_chars.saturating_sub(budget);
    }

    let ellipsis_chars =
        usize::from(start > 0) + usize::from(end < total_chars);
    budget = PREVIEW_MAX_CHARS.saturating_sub(ellipsis_chars);
    start = center.saturating_sub(budget / 2);
    end = start.saturating_add(budget).min(total_chars);
    if end == total_chars {
        start = total_chars.saturating_sub(budget);
    }

    (start, end)
}

/// Cuts a preview range and adds ellipses.
fn cut_preview(
    text: &str,
    total_chars: usize,
    start: usize,
    end: usize,
) -> String {
    let (start, end) = snap_preview_range(text, total_chars, start, end);
    let mut preview = String::new();
    if start > 0 {
        preview.push(ELLIPSIS);
    }
    preview.extend(text.chars().skip(start).take(end.saturating_sub(start)));
    if end < total_chars {
        preview.push(ELLIPSIS);
    }

    debug_assert!(preview.chars().count() <= PREVIEW_MAX_CHARS);
    preview
}

/// Snaps a preview range to word boundaries where possible.
fn snap_preview_range(
    text: &str,
    total_chars: usize,
    start: usize,
    end: usize,
) -> (usize, usize) {
    let snapped_start = snap_preview_start(text, start, end);
    let snapped_end = snap_preview_end(text, total_chars, snapped_start, end);
    if snapped_start < snapped_end {
        return (snapped_start, snapped_end);
    }

    (start, end)
}

/// Snaps a preview start to a word boundary.
fn snap_preview_start(text: &str, start: usize, end: usize) -> usize {
    if start == 0
        || char_at(text, start.saturating_sub(1))
            .is_some_and(|ch| ch.is_whitespace())
    {
        return start;
    }

    text.chars()
        .enumerate()
        .skip(start)
        .take(end.saturating_sub(start))
        .find(|(_index, character)| character.is_whitespace())
        .map_or(start, |(index, _character)| index + 1)
}

/// Snaps a preview end to a word boundary.
fn snap_preview_end(
    text: &str,
    total_chars: usize,
    start: usize,
    end: usize,
) -> usize {
    if end == total_chars
        || char_at(text, end).is_some_and(|ch| ch.is_whitespace())
    {
        return end;
    }

    let mut boundary = None;
    for (index, character) in text
        .chars()
        .enumerate()
        .skip(start)
        .take(end.saturating_sub(start))
    {
        if character.is_whitespace() {
            boundary = Some(index);
        }
    }

    boundary.unwrap_or(end)
}

/// Returns a character at a char index.
fn char_at(text: &str, index: usize) -> Option<char> {
    text.chars().nth(index)
}

/// Orders candidate keys for stable results.
fn compare_keys(
    left: &CandidateKey,
    right: &CandidateKey,
) -> std::cmp::Ordering {
    left.collection
        .cmp(&right.collection)
        .then_with(|| left.record_id.cmp(&right.record_id))
}

/// Orders search candidates for stable results.
fn compare_candidates(
    left: &SearchCandidate,
    right: &SearchCandidate,
) -> std::cmp::Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.collection.cmp(&right.collection))
        .then_with(|| left.record_id.cmp(&right.record_id))
}

/// Returns a storage data root.
fn data_root(path: Option<&Path>) -> Result<(PathBuf, Option<TempDir>)> {
    if let Some(path) = path {
        return Ok((path.to_path_buf(), None));
    }

    let temp_dir =
        tempfile::tempdir().context("failed to create temp index root")?;
    Ok((temp_dir.path().to_path_buf(), Some(temp_dir)))
}

/// Validates an index name.
fn validate_name(name: &str) -> std::result::Result<(), IndexError> {
    if name.is_empty() {
        return Err(IndexError::InvalidInput {
            message: "index name must not be empty".to_string(),
        });
    }
    if name.contains('/') || name.contains('\\') {
        return Err(IndexError::InvalidInput {
            message: "index name must not contain path separators".to_string(),
        });
    }

    let mut components = Path::new(name).components();
    let valid = matches!(
        (components.next(), components.next()),
        (Some(Component::Normal(part)), None) if part == OsStr::new(name)
    );
    if valid {
        return Ok(());
    }

    Err(IndexError::InvalidInput {
        message: "index name must be a plain path segment".to_string(),
    })
}

/// Builds graph labels for a record collection.
fn record_labels(collection: &str) -> Vec<String> {
    vec![
        RECORD_LABEL.to_string(),
        graph_token(COLLECTION_LABEL_PREFIX, collection),
    ]
}

/// Borrows graph labels as string slices.
fn label_refs(labels: &[String]) -> Vec<&str> {
    labels.iter().map(String::as_str).collect()
}

/// Checks graph label identity without depending on label order.
fn labels_match(left: &[String], right: &[String]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .all(|label| right.iter().any(|item| item == label))
}

/// Builds a graph node for a record.
fn graph_node(record: &Record) -> Result<GraphNode> {
    Ok(GraphNode {
        id: record.id.clone(),
        labels: record_labels(&record.collection),
        name: record.title.clone(),
        description: None,
        metadata: Some(graph_payload(record)?),
    })
}

/// Builds graph payload metadata.
fn graph_payload(record: &Record) -> Result<serde_json::Value> {
    let mut values = serde_json::Map::new();
    values.insert(
        GRAPH_METADATA_KEY.to_string(),
        serde_json::Value::String(
            serde_json::to_string(&record.metadata)
                .context("failed to serialize graph metadata")?,
        ),
    );

    Ok(serde_json::Value::Object(values))
}

/// Encodes arbitrary public values as graph-safe identifiers.
fn graph_token(prefix: &str, value: &str) -> String {
    format!("{prefix}{}", crate::hex::encode(value.as_bytes()))
}

/// Decodes graph-safe identifiers back into public values.
fn graph_value(prefix: &str, token: &str) -> Option<String> {
    let bytes = crate::hex::decode(token.strip_prefix(prefix)?)?;
    String::from_utf8(bytes).ok()
}

/// Returns the public collection label.
fn collection_from_labels(labels: &[String]) -> Option<String> {
    labels
        .iter()
        .find_map(|label| graph_value(COLLECTION_LABEL_PREFIX, label))
}

/// Returns the public relationship predicate.
fn relationship_predicate(predicate: &str) -> String {
    graph_value(RELATIONSHIP_TYPE_PREFIX, predicate)
        .unwrap_or_else(|| predicate.to_string())
}

/// Reads outgoing relationships for one record.
fn relationships(
    graph: &dyn GraphDbContext,
    collection: &str,
    record_id: &str,
) -> Result<Vec<RecordRelationship>> {
    let labels = record_labels(collection);
    let label_refs = label_refs(&labels);
    let mut relationships = graph
        .neighbors(&label_refs, record_id)
        .with_context(|| {
            format!(
                "failed to read relationships for record '{record_id}' in collection '{collection}'",
            )
        })?
        .into_iter()
        .filter(|(edge, _node)| {
            edge.source == record_id && labels_match(&edge.source_labels, &labels)
        })
        .map(|(edge, _node)| {
            let collection = collection_from_labels(&edge.target_labels)
                .context("relationship target missing collection label")?;
            Ok(RecordRelationship {
                predicate: relationship_predicate(&edge.predicate),
                record_id: edge.target,
                collection,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    relationships.sort_by(|left, right| {
        (&left.collection, &left.record_id, &left.predicate).cmp(&(
            &right.collection,
            &right.record_id,
            &right.predicate,
        ))
    });

    Ok(relationships)
}

/// Deletes all edges touching a record.
fn delete_touching_edges(
    graph: &dyn GraphDbContext,
    labels: &[&str],
    record_id: &str,
) -> Result<()> {
    for (edge, _node) in graph.neighbors(labels, record_id)? {
        delete_edge_absorb_missing(graph, &edge)?;
    }

    Ok(())
}

/// Deletes an edge and ignores missing edges.
fn delete_edge_absorb_missing(
    graph: &dyn GraphDbContext,
    edge: &GraphEdge,
) -> Result<()> {
    match graph.delete_edge(edge) {
        Ok(()) | Err(GraphError::NotFound { .. }) => Ok(()),
        Err(error) => Err(error).context("failed to delete graph edge"),
    }
}

/// Deletes a node and ignores missing nodes.
fn delete_node_absorb_missing(
    graph: &dyn GraphDbContext,
    labels: &[&str],
    record_id: &str,
) -> Result<()> {
    match graph.delete_node(labels, record_id) {
        Ok(()) | Err(GraphError::NotFound { .. }) => Ok(()),
        Err(error) => Err(error).context("failed to delete graph node"),
    }
}
