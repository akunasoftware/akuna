use std::{
    collections::HashMap,
    fs,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use super::{
    CandidateEvidence, EvidenceRef, PREVIEW_MAX_CHARS, RankingCandidate,
    SearchCandidate, apply_rerank_scores, build_preview, candidate_preview,
    expand_candidates, graph_node, record_labels, relationships,
};
use crate::embedding::EmbeddingModel;
use crate::index::{
    Index, IndexError, IndexOptions, IndexSearchQuery, Metadata,
    MetadataFilter, MetadataValue, Record, RecordRelationship,
};
use crate::storage::vector::{
    TextSearchQuery, VectorDbContext, VectorError, VectorSearchQuery,
    VectorTextIndex,
};
use crate::storage::{
    ChunkEntry, ChunkSearchResult, GraphDbContext, GraphEdge, GraphError,
    GraphNode, RecordEntry, RecordSearchResult, in_memory_context,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Builds index options for tests.
fn test_options(path: Option<std::path::PathBuf>) -> IndexOptions {
    IndexOptions {
        path,
        embedding_model: EmbeddingModel::MiniLmL6,
        reranking_model: None,
        ..Default::default()
    }
}

/// Builds a test record.
fn record(
    id: &str,
    title: &str,
    content: &str,
    relationships: Vec<RecordRelationship>,
) -> Record {
    record_in("docs", id, title, content, Metadata::new(), relationships)
}

/// Builds a test record in a collection.
fn record_in(
    collection: &str,
    id: &str,
    title: &str,
    content: &str,
    metadata: Metadata,
    relationships: Vec<RecordRelationship>,
) -> Record {
    Record {
        id: id.to_string(),
        collection: collection.to_string(),
        title: title.to_string(),
        content: content.to_string(),
        metadata,
        relationships,
    }
}

/// Builds test metadata.
fn metadata(values: &[(&str, MetadataValue)]) -> Metadata {
    values
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect()
}

/// Builds a test relationship.
fn relationship(record_id: &str) -> RecordRelationship {
    RecordRelationship {
        predicate: "cites".to_string(),
        record_id: record_id.to_string(),
        collection: "docs".to_string(),
    }
}

/// Builds a text search query.
fn text_query(text: &str) -> TextSearchQuery {
    TextSearchQuery {
        text: text.to_string(),
        collections: vec!["docs".to_string()],
        filter: None,
        limit: 10,
    }
}

/// Builds an index search query.
fn search_query(text: &str) -> IndexSearchQuery {
    IndexSearchQuery {
        text: text.to_string(),
        collections: vec!["docs".to_string()],
        filter: None,
        limit: 10,
    }
}

/// Builds a search candidate from a record.
fn search_candidate(record: &Record, score: f32) -> SearchCandidate {
    SearchCandidate {
        record_id: record.id.clone(),
        collection: record.collection.clone(),
        title: record.title.clone(),
        content: record.content.clone(),
        metadata: record.metadata.clone(),
        score,
        best_evidence: CandidateEvidence::Title,
    }
}

/// Returns candidate record ids.
fn candidate_ids(candidates: &[SearchCandidate]) -> Vec<&str> {
    candidates
        .iter()
        .map(|candidate| candidate.record_id.as_str())
        .collect()
}

/// Builds a vector record entry from a record.
fn record_entry(record: &Record) -> RecordEntry {
    RecordEntry {
        record_id: record.id.clone(),
        collection: record.collection.clone(),
        title: record.title.clone(),
        title_embedding: Vec::new(),
        content: record.content.clone(),
        metadata: record.metadata.clone(),
    }
}

/// Builds a graph edge between records.
fn graph_edge(source: &Record, target: &Record) -> GraphEdge {
    GraphEdge {
        source_labels: record_labels(&source.collection),
        source: source.id.clone(),
        predicate: "cites".to_string(),
        target: target.id.clone(),
        target_labels: record_labels(&target.collection),
    }
}

/// Stores record graph nodes.
fn put_graph_records(
    graph: &dyn crate::storage::GraphDbContext,
    records: &[Record],
) -> TestResult {
    for record in records {
        graph.put_node(&graph_node(record)?)?;
    }

    Ok(())
}

struct TestVector {
    records: HashMap<(String, String), RecordEntry>,
    record_failure: Option<Arc<RecordFailure>>,
}

struct RecordFailure {
    chunks: AtomicUsize,
    deletes: AtomicUsize,
    writes: AtomicUsize,
}

impl TestVector {
    /// Builds vector storage from records.
    fn new(records: &[Record]) -> Self {
        Self {
            records: records
                .iter()
                .map(|record| {
                    (
                        (record.collection.clone(), record.id.clone()),
                        record_entry(record),
                    )
                })
                .collect(),
            record_failure: None,
        }
    }

    /// Builds vector storage that rejects record rows after accepting chunks.
    fn failing_record_write(record_failure: Arc<RecordFailure>) -> Self {
        Self {
            records: HashMap::new(),
            record_failure: Some(record_failure),
        }
    }
}

struct FailingGraph {
    inner: Box<dyn GraphDbContext>,
    fail_node: Option<String>,
    fail_delete: Arc<AtomicBool>,
}

impl FailingGraph {
    /// Builds a graph wrapper that can fail one node write or delete.
    fn new(
        inner: Box<dyn GraphDbContext>,
        fail_node: Option<String>,
        fail_delete: Arc<AtomicBool>,
    ) -> Self {
        Self {
            inner,
            fail_node,
            fail_delete,
        }
    }
}

impl GraphDbContext for FailingGraph {
    fn put_node(&self, node: &GraphNode) -> Result<(), GraphError> {
        if self.fail_node.as_deref() == Some(node.id.as_str()) {
            return Err(graph_failure());
        }

        self.inner.put_node(node)
    }

    fn get_node(
        &self,
        labels: &[&str],
        id: &str,
    ) -> Result<Option<GraphNode>, GraphError> {
        self.inner.get_node(labels, id)
    }

    fn delete_node(&self, labels: &[&str], id: &str) -> Result<(), GraphError> {
        if self.fail_delete.load(Ordering::Relaxed) {
            return Err(graph_failure());
        }

        self.inner.delete_node(labels, id)
    }

    fn put_edge(&self, edge: &GraphEdge) -> Result<(), GraphError> {
        self.inner.put_edge(edge)
    }

    fn delete_edge(&self, edge: &GraphEdge) -> Result<(), GraphError> {
        self.inner.delete_edge(edge)
    }

    fn neighbors(
        &self,
        labels: &[&str],
        id: &str,
    ) -> Result<Vec<(GraphEdge, GraphNode)>, GraphError> {
        self.inner.neighbors(labels, id)
    }
}

/// Builds a graph engine failure.
fn graph_failure() -> GraphError {
    GraphError::QueryExecution {
        engine: "test",
        source: Box::new(std::io::Error::other("graph failure")),
    }
}

#[async_trait::async_trait]
impl VectorDbContext for TestVector {
    async fn put_record_with_chunks(
        &self,
        _record: &RecordEntry,
        chunks: &[ChunkEntry],
    ) -> Result<(), VectorError> {
        if let Some(record_failure) = &self.record_failure {
            record_failure.chunks.store(chunks.len(), Ordering::Relaxed);
            record_failure.writes.fetch_add(1, Ordering::Relaxed);
            return Err(VectorError::InvalidDimensions { dimensions: 0 });
        }

        Ok(())
    }

    async fn put_chunks(
        &self,
        _collection: &str,
        _record_id: &str,
        chunks: &[ChunkEntry],
    ) -> Result<(), VectorError> {
        if let Some(record_failure) = &self.record_failure {
            record_failure.chunks.store(chunks.len(), Ordering::Relaxed);
            record_failure.writes.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    async fn put_record(
        &self,
        _record: &RecordEntry,
    ) -> Result<(), VectorError> {
        if self.record_failure.is_some() {
            return Err(VectorError::InvalidDimensions { dimensions: 0 });
        }

        Ok(())
    }

    async fn delete_record(
        &self,
        _collection: &str,
        _record_id: &str,
    ) -> Result<(), VectorError> {
        if let Some(record_failure) = &self.record_failure {
            record_failure.chunks.store(0, Ordering::Relaxed);
            record_failure.deletes.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    async fn get_record(
        &self,
        collection: &str,
        record_id: &str,
    ) -> Result<Option<RecordEntry>, VectorError> {
        Ok(self
            .records
            .get(&(collection.to_string(), record_id.to_string()))
            .cloned())
    }

    async fn get_records(
        &self,
        keys: &[(String, String)],
    ) -> Result<Vec<RecordEntry>, VectorError> {
        Ok(keys
            .iter()
            .filter_map(|key| self.records.get(key).cloned())
            .collect())
    }

    async fn search_chunks(
        &self,
        _query: &VectorSearchQuery,
    ) -> Result<Vec<ChunkSearchResult>, VectorError> {
        Ok(Vec::new())
    }

    async fn search_titles(
        &self,
        _query: &VectorSearchQuery,
    ) -> Result<Vec<RecordSearchResult>, VectorError> {
        Ok(Vec::new())
    }

    async fn search_chunks_text(
        &self,
        _query: &TextSearchQuery,
    ) -> Result<Vec<ChunkSearchResult>, VectorError> {
        Ok(Vec::new())
    }

    async fn search_titles_text(
        &self,
        _query: &TextSearchQuery,
    ) -> Result<Vec<RecordSearchResult>, VectorError> {
        Ok(Vec::new())
    }
}

/// Extracts an expected index error without requiring `T: Debug`.
fn expect_error<T>(result: Result<T, IndexError>, message: &str) -> IndexError {
    match result {
        Ok(_) => panic!("{message}"),
        Err(error) => error,
    }
}

#[tokio::test]
async fn crud_upsert_remove_persist() -> TestResult {
    let temp_dir = tempfile::tempdir()?;
    let options = test_options(Some(temp_dir.path().to_path_buf()));
    let index = Index::new(options.clone()).await?;
    let alpha = record(
        "alpha",
        "Alpha",
        "oldtoken body",
        vec![relationship("beta")],
    );
    let beta = record("beta", "Beta", "beta body", Vec::new());

    index.add(vec![alpha.clone(), beta.clone()]).await?;

    let alpha_read = index.get("docs", "alpha").await?.expect("alpha stored");
    assert_eq!(alpha_read.title, "Alpha");
    assert_eq!(alpha_read.relationships, vec![relationship("beta")]);
    assert_eq!(
        index
            .vector
            .search_chunks_text(&text_query("oldtoken"))
            .await?
            .len(),
        1,
    );

    index
        .add(vec![record(
            "alpha",
            "Alpha 2",
            "newtoken body",
            Vec::new(),
        )])
        .await?;

    let alpha_read = index.get("docs", "alpha").await?.expect("alpha stored");
    assert_eq!(alpha_read.title, "Alpha 2");
    assert!(alpha_read.relationships.is_empty());
    assert!(
        index
            .vector
            .search_chunks_text(&text_query("oldtoken"))
            .await?
            .is_empty()
    );
    assert_eq!(
        index
            .vector
            .search_chunks_text(&text_query("newtoken"))
            .await?
            .len(),
        1,
    );

    index
        .add(vec![record(
            "beta",
            "Beta",
            "beta body",
            vec![relationship("alpha")],
        )])
        .await?;
    index.remove("docs", "alpha").await?;
    assert!(index.get("docs", "alpha").await?.is_none());
    assert!(
        index
            .get("docs", "beta")
            .await?
            .expect("beta stored")
            .relationships
            .is_empty()
    );
    index.remove("docs", "missing").await?;

    drop(index);
    let reopened = Index::new(options.clone()).await?;
    assert!(reopened.get("docs", "beta").await?.is_some());

    drop(reopened);
    let error = expect_error(
        Index::new(IndexOptions {
            fulltext: false,
            ..options
        })
        .await,
        "manifest mismatch should fail",
    );
    assert!(error.to_string().contains("fulltext"));

    Ok(())
}

#[tokio::test]
async fn graph_false_fulltext_false() -> TestResult {
    let index = Index::new(IndexOptions {
        graph: false,
        fulltext: false,
        reranking_model: None,
        embedding_model: EmbeddingModel::MiniLmL6,
        ..Default::default()
    })
    .await?;

    let error = index
        .add(vec![record(
            "related",
            "Related",
            "related body",
            vec![relationship("target")],
        )])
        .await
        .expect_err("relationships should require graph");
    assert!(
        error
            .to_string()
            .contains("relationships require graph storage")
    );
    assert!(index.get("docs", "related").await?.is_none());

    index
        .add(vec![record("solo", "Solo", "solo body", Vec::new())])
        .await?;
    assert!(
        index
            .get("docs", "solo")
            .await?
            .expect("solo stored")
            .relationships
            .is_empty()
    );
    assert!(matches!(
        index.vector.search_chunks_text(&text_query("solo")).await,
        Err(VectorError::FullTextDisabled {
            target: VectorTextIndex::Chunks,
        }),
    ));
    let audit = index.audit_records();
    let record = audit.last().expect("add should leave an audit record");
    assert_eq!(record.engine, "index");
    assert_eq!(record.operation, super::IndexAuditOperation::Add);
    assert_eq!(record.outputs.get("records"), Some(&1));

    Ok(())
}

#[tokio::test]
async fn record_write_failure_rolls_back() -> TestResult {
    let record_failure = Arc::new(RecordFailure {
        chunks: AtomicUsize::new(0),
        deletes: AtomicUsize::new(0),
        writes: AtomicUsize::new(0),
    });
    let mut options = test_options(None);
    options.graph = false;
    let Index {
        embedder,
        reranker,
        chunking,
        ..
    } = Index::new(options).await?;
    let index = Index::with_dependencies(
        Box::new(TestVector::failing_record_write(record_failure.clone())),
        None,
        embedder,
        reranker,
        false,
        chunking,
    );

    let error = index
        .add(vec![record("record", "Record", "body", Vec::new())])
        .await
        .expect_err("vector write should fail");
    assert!(matches!(error, IndexError::Write { .. }));
    assert!(std::error::Error::source(&error).is_some());
    assert_eq!(record_failure.writes.load(Ordering::Relaxed), 1);
    assert_eq!(record_failure.deletes.load(Ordering::Relaxed), 1);
    assert_eq!(record_failure.chunks.load(Ordering::Relaxed), 0);

    Ok(())
}

#[tokio::test]
async fn graph_write_and_remove_failures_roll_back() -> TestResult {
    let temp_dir = tempfile::tempdir()?;
    let Index {
        vector,
        graph,
        embedder,
        reranker,
        fulltext,
        chunking,
        ..
    } = Index::new(test_options(Some(temp_dir.path().to_path_buf()))).await?;
    let graph = graph.expect("graph should be enabled");
    let fail_delete = Arc::new(AtomicBool::new(false));
    let index = Index::with_dependencies(
        vector,
        Some(Box::new(FailingGraph::new(
            graph,
            Some("fail".to_string()),
            fail_delete.clone(),
        ))),
        embedder,
        reranker,
        fulltext,
        chunking,
    );

    assert!(
        index
            .add(vec![
                record("first", "First", "first body", Vec::new()),
                record("fail", "Fail", "fail body", Vec::new()),
            ])
            .await
            .is_err()
    );
    assert!(index.get("docs", "first").await?.is_none());

    index
        .add(vec![
            record(
                "source",
                "Source",
                "source body",
                vec![relationship("target")],
            ),
            record("target", "Target", "target body", Vec::new()),
        ])
        .await?;
    fail_delete.store(true, Ordering::Relaxed);

    assert!(index.remove("docs", "source").await.is_err());
    assert_eq!(
        index
            .get("docs", "source")
            .await?
            .expect("source should be restored")
            .relationships,
        vec![relationship("target")],
    );

    Ok(())
}

#[tokio::test]
async fn manifest_and_name_errors() -> TestResult {
    let invalid_name = expect_error(
        Index::new(IndexOptions {
            name: "../bad".to_string(),
            reranking_model: None,
            ..Default::default()
        })
        .await,
        "invalid name should fail",
    );
    assert!(invalid_name.to_string().contains("path separators"));

    let temp_dir = tempfile::tempdir()?;
    let root = temp_dir.path().join("default");
    fs::create_dir_all(root.join("vector"))?;
    let missing = expect_error(
        Index::new(test_options(Some(temp_dir.path().to_path_buf()))).await,
        "missing manifest should fail",
    );
    assert!(missing.to_string().contains("manifest missing"));

    let temp_dir = tempfile::tempdir()?;
    let root = temp_dir.path().join("default");
    fs::create_dir_all(&root)?;
    fs::write(root.join("manifest.yaml"), "not yaml")?;
    let corrupt = expect_error(
        Index::new(test_options(Some(temp_dir.path().to_path_buf()))).await,
        "corrupt manifest should fail",
    );
    assert!(corrupt.to_string().contains("manifest corrupt"));

    Ok(())
}

#[tokio::test]
async fn missing_relationship_target_preserves_existing_edges() -> TestResult {
    let index = Index::new(test_options(None)).await?;
    let source = record(
        "source",
        "Source",
        "source body",
        vec![relationship("target")],
    );
    let target = record("target", "Target", "target body", Vec::new());
    index.add(vec![source, target]).await?;

    let error = index
        .add(vec![record(
            "source",
            "Source",
            "source body",
            vec![relationship("missing")],
        )])
        .await
        .expect_err("missing target should fail");
    assert!(error.to_string().contains("relationship target not found"));
    assert_eq!(
        index
            .get("docs", "source")
            .await?
            .expect("source should remain")
            .relationships,
        vec![relationship("target")],
    );
    let unrelated =
        record("unrelated", "Unrelated", "unrelated body", Vec::new());
    let error = index
        .add(vec![
            unrelated.clone(),
            record(
                "new-source",
                "New Source",
                "new source body",
                vec![relationship("missing")],
            ),
        ])
        .await
        .expect_err("batch preflight should fail");
    assert!(matches!(error, IndexError::InvalidInput { .. }));
    assert!(index.get("docs", &unrelated.id).await?.is_none());

    Ok(())
}

#[tokio::test]
async fn add_accepts_public_graph_strings() -> TestResult {
    let index = Index::new(test_options(None)).await?;
    let source = record_in(
        "project-notes",
        "source",
        "Source",
        "body",
        metadata(&[("file.path", MetadataValue::Text("a/b.md".to_string()))]),
        vec![RecordRelationship {
            predicate: "related-to".to_string(),
            record_id: "target".to_string(),
            collection: "mime-type".to_string(),
        }],
    );
    let target = record_in(
        "mime-type",
        "target",
        "Target",
        "body",
        Metadata::new(),
        Vec::new(),
    );

    index.add(vec![source.clone(), target]).await?;

    let stored = index
        .get("project-notes", "source")
        .await?
        .expect("source stored");
    assert_eq!(stored.metadata, source.metadata);
    assert_eq!(stored.relationships, source.relationships);

    Ok(())
}

#[tokio::test]
async fn search_title_limit_empty() -> TestResult {
    let index = Index::new(IndexOptions {
        fulltext: false,
        graph: false,
        ..test_options(None)
    })
    .await?;
    index
        .add(vec![record(
            "title",
            "Volcano Needle Title",
            "plain body",
            Vec::new(),
        )])
        .await?;

    let results = index.search(search_query("Volcano Needle Title")).await?;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].record_id, "title");
    assert_eq!(results[0].preview.as_deref(), Some("plain body"));

    assert!(
        index
            .search(IndexSearchQuery {
                limit: 0,
                ..search_query("Volcano Needle Title")
            })
            .await?
            .is_empty()
    );
    let error = expect_error(
        index.search(search_query("   ")).await,
        "empty query should fail",
    );
    assert!(error.to_string().contains("search text must not be empty"));

    Ok(())
}

#[tokio::test]
async fn search_previews_hits_and_expansions() -> TestResult {
    let index = Index::new(test_options(None)).await?;
    let filler = (0..90)
        .map(|index| format!("filler{index}"))
        .collect::<Vec<_>>()
        .join(" ");
    let passage = "volcano needle passage";
    let content = format!("{filler} {passage} {filler}");
    index
        .add(vec![
            record("source", "Source", &content, vec![relationship("target")]),
            record(
                "target",
                "Target",
                "expanded target leading body",
                Vec::new(),
            ),
        ])
        .await?;

    let results = index
        .search(IndexSearchQuery {
            text: "volcano needle".to_string(),
            limit: 2,
            ..search_query("volcano needle")
        })
        .await?;

    let Some(source) =
        results.iter().find(|result| result.record_id == "source")
    else {
        panic!("source result missing");
    };
    let Some(source_preview) = source.preview.as_ref() else {
        panic!("source preview missing");
    };
    assert!(source_preview.contains(passage));
    assert!(source_preview.chars().count() <= PREVIEW_MAX_CHARS);

    let Some(target) =
        results.iter().find(|result| result.record_id == "target")
    else {
        panic!("target expansion missing");
    };
    assert_eq!(
        target.preview.as_deref(),
        Some("expanded target leading body"),
    );

    Ok(())
}

#[tokio::test]
async fn search_filters_collection_metadata() -> TestResult {
    let index = Index::new(test_options(None)).await?;
    let red = metadata(&[("color", MetadataValue::Text("red".to_string()))]);
    let blue = metadata(&[("color", MetadataValue::Text("blue".to_string()))]);
    index
        .add(vec![
            record_in(
                "docs",
                "red-doc",
                "Red Doc",
                "filterneedle body",
                red.clone(),
                Vec::new(),
            ),
            record_in(
                "docs",
                "blue-doc",
                "Blue Doc",
                "filterneedle body",
                blue,
                Vec::new(),
            ),
            record_in(
                "notes",
                "red-note",
                "Red Note",
                "filterneedle body",
                red,
                Vec::new(),
            ),
        ])
        .await?;

    let results = index
        .search(IndexSearchQuery {
            text: "filterneedle".to_string(),
            collections: vec!["docs".to_string()],
            filter: Some(MetadataFilter::Equals {
                key: "color".to_string(),
                value: MetadataValue::Text("red".to_string()),
            }),
            limit: 10,
        })
        .await?;

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].record_id, "red-doc");
    assert_eq!(results[0].collection, "docs");

    Ok(())
}

#[tokio::test]
async fn search_expands_relationship() -> TestResult {
    let graph = in_memory_context();
    let source = record("source", "Source", "source body", Vec::new());
    let target = record("target", "Target", "target body", Vec::new());
    put_graph_records(graph.as_ref(), &[source.clone(), target.clone()])?;
    graph.put_edge(&graph_edge(&source, &target))?;
    graph.put_edge(&graph_edge(&source, &source))?;
    graph.put_edge(&graph_edge(&target, &source))?;
    let vector = TestVector::new(std::slice::from_ref(&target));

    let results = expand_candidates(
        Some(graph.as_ref()),
        &vector,
        None,
        "source",
        &IndexSearchQuery {
            limit: 2,
            ..search_query("source")
        },
        vec![search_candidate(&source, 1.0)],
    )
    .await?;

    assert_eq!(results.len(), 2);
    let Some(expanded) =
        results.iter().find(|result| result.record_id == "target")
    else {
        panic!("target expansion missing");
    };
    assert_eq!(expanded.score, 0.5);
    assert_eq!(
        expanded.best_evidence,
        CandidateEvidence::LeadingWindow("target body".to_string()),
    );

    Ok(())
}

#[test]
fn relationships_scope_source_collection() -> TestResult {
    let graph = in_memory_context();
    let docs = record("same", "Docs", "docs body", Vec::new());
    let notes = record_in(
        "notes",
        "same",
        "Notes",
        "notes body",
        Metadata::new(),
        Vec::new(),
    );
    put_graph_records(graph.as_ref(), &[docs.clone(), notes.clone()])?;
    graph.put_edge(&GraphEdge {
        source_labels: record_labels("notes"),
        source: "same".to_string(),
        predicate: "cites".to_string(),
        target: "same".to_string(),
        target_labels: record_labels("docs"),
    })?;

    assert!(relationships(graph.as_ref(), "docs", "same")?.is_empty());
    assert_eq!(
        relationships(graph.as_ref(), "notes", "same")?,
        vec![RecordRelationship {
            predicate: "cites".to_string(),
            record_id: "same".to_string(),
            collection: "docs".to_string(),
        }],
    );

    Ok(())
}

#[tokio::test]
async fn search_expansion_filters_scope() -> TestResult {
    let graph = in_memory_context();
    let red = metadata(&[("color", MetadataValue::Text("red".to_string()))]);
    let blue = metadata(&[("color", MetadataValue::Text("blue".to_string()))]);
    let source = record("source", "Source", "source body", Vec::new());
    let blue_target =
        record_in("docs", "blue", "Blue", "blue body", blue, Vec::new());
    let note_target = record_in(
        "notes",
        "red-note",
        "Red Note",
        "note body",
        red,
        Vec::new(),
    );
    put_graph_records(
        graph.as_ref(),
        &[source.clone(), blue_target.clone(), note_target.clone()],
    )?;
    graph.put_edge(&graph_edge(&source, &blue_target))?;
    graph.put_edge(&graph_edge(&source, &note_target))?;
    let vector = TestVector::new(&[blue_target, note_target]);

    let results = expand_candidates(
        Some(graph.as_ref()),
        &vector,
        None,
        "source",
        &IndexSearchQuery {
            filter: Some(MetadataFilter::Equals {
                key: "color".to_string(),
                value: MetadataValue::Text("red".to_string()),
            }),
            limit: 2,
            ..search_query("source")
        },
        vec![search_candidate(&source, 1.0)],
    )
    .await?;

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].record_id, "source");

    Ok(())
}

#[tokio::test]
async fn search_expansion_filters_collection_metadata_key() -> TestResult {
    let graph = in_memory_context();
    let source = record("source", "Source", "source body", Vec::new());
    let target = record_in(
        "docs",
        "target",
        "Target",
        "target body",
        metadata(&[("file.path", MetadataValue::Text("custom".to_string()))]),
        Vec::new(),
    );
    put_graph_records(graph.as_ref(), &[source.clone(), target.clone()])?;
    graph.put_edge(&graph_edge(&source, &target))?;
    let vector = TestVector::new(std::slice::from_ref(&target));

    let results = expand_candidates(
        Some(graph.as_ref()),
        &vector,
        None,
        "source",
        &IndexSearchQuery {
            filter: Some(MetadataFilter::Equals {
                key: "file.path".to_string(),
                value: MetadataValue::Text("custom".to_string()),
            }),
            limit: 2,
            ..search_query("source")
        },
        vec![search_candidate(&source, 1.0)],
    )
    .await?;

    assert!(results.iter().any(|result| result.record_id == "target"));

    Ok(())
}

#[tokio::test]
async fn search_expansion_graph_disabled_or_empty_is_inert() -> TestResult {
    let vector = TestVector::new(&[]);
    let first = record("first", "First", "first body", Vec::new());
    let second = record("second", "Second", "second body", Vec::new());
    let candidates = vec![
        search_candidate(&first, 0.2),
        search_candidate(&second, 0.1),
    ];

    let disabled = expand_candidates(
        None,
        &vector,
        None,
        "first",
        &search_query("first"),
        candidates.clone(),
    )
    .await?;
    assert_eq!(candidate_ids(&disabled), vec!["first", "second"]);

    let graph = in_memory_context();
    put_graph_records(graph.as_ref(), &[first, second])?;
    let empty = expand_candidates(
        Some(graph.as_ref()),
        &vector,
        None,
        "first",
        &search_query("first"),
        candidates,
    )
    .await?;
    assert_eq!(candidate_ids(&empty), vec!["first", "second"]);
    assert_eq!(empty[0].score, 0.2);
    assert_eq!(empty[1].score, 0.1);

    Ok(())
}

#[tokio::test]
async fn search_expansion_caps_deterministically() -> TestResult {
    let graph = in_memory_context();
    let source = record("source", "Source", "source body", Vec::new());
    let targets = ["z", "d", "a", "c", "b", "e"]
        .into_iter()
        .map(|id| record(id, id, "target body", Vec::new()))
        .collect::<Vec<_>>();
    let mut records = vec![source.clone()];
    records.extend(targets.clone());
    put_graph_records(graph.as_ref(), &records)?;
    for target in targets.iter().rev() {
        graph.put_edge(&graph_edge(&source, target))?;
    }
    let vector = TestVector::new(&targets);

    let results = expand_candidates(
        Some(graph.as_ref()),
        &vector,
        None,
        "source",
        &IndexSearchQuery {
            limit: 2,
            ..search_query("source")
        },
        vec![search_candidate(&source, 1.0)],
    )
    .await?;

    assert_eq!(candidate_ids(&results), vec!["source", "a", "b", "c", "d"]);

    Ok(())
}

#[test]
fn rerank_scores_select_title_evidence() -> TestResult {
    let candidates = vec![RankingCandidate {
        record_id: "record".to_string(),
        collection: "docs".to_string(),
        title: "strong title".to_string(),
        content: "body".to_string(),
        metadata: Metadata::new(),
        fused_score: 0.1,
        title_hit: true,
        chunks: vec!["weak chunk".to_string()],
        best_chunk: None,
    }];
    let evidence_refs = vec![
        EvidenceRef {
            candidate_index: 0,
            evidence: CandidateEvidence::Chunk("weak chunk".to_string()),
        },
        EvidenceRef {
            candidate_index: 0,
            evidence: CandidateEvidence::Title,
        },
    ];

    let results =
        apply_rerank_scores(candidates, evidence_refs, vec![-10.0, 10.0])?;
    assert_eq!(results.len(), 1);
    assert!(results[0].score > 0.99);
    assert_eq!(results[0].best_evidence, CandidateEvidence::Title);

    Ok(())
}

#[test]
fn preview_collapses_short_evidence() {
    assert_eq!(
        build_preview("alpha beta", " alpha\n\n beta\t gamma ").as_deref(),
        Some("alpha beta gamma"),
    );
}

#[test]
fn preview_centers_late_passage() {
    let prefix = (0..90)
        .map(|index| format!("before{index}"))
        .collect::<Vec<_>>()
        .join(" ");
    let suffix = (0..90)
        .map(|index| format!("after{index}"))
        .collect::<Vec<_>>()
        .join(" ");
    let passage = "volcano needle passage";
    let evidence = format!("{prefix} {passage} {suffix}");

    let Some(preview) = build_preview("volcano needle", &evidence) else {
        panic!("preview missing");
    };

    assert!(preview.starts_with('…'));
    assert!(preview.ends_with('…'));
    assert!(preview.contains(passage));
    assert!(!preview.contains("before0 before1"));
    assert!(preview.chars().count() <= PREVIEW_MAX_CHARS);
}

#[test]
fn preview_zero_overlap_uses_head() {
    let evidence = (0..90)
        .map(|index| format!("filler{index}"))
        .collect::<Vec<_>>()
        .join(" ");

    let Some(preview) = build_preview("missing", &evidence) else {
        panic!("preview missing");
    };

    assert!(preview.starts_with("filler0 filler1"));
    assert!(preview.ends_with('…'));
    assert!(preview.chars().count() <= PREVIEW_MAX_CHARS);
}

#[test]
fn preview_unicode_safe() {
    let filler = "🙂 ".repeat(140);
    let evidence = format!("{filler}cafés needle passage {filler}");

    let Some(preview) = build_preview("CAFÉS needle", &evidence) else {
        panic!("preview missing");
    };

    assert!(preview.contains("cafés needle passage"));
    assert!(preview.chars().count() <= PREVIEW_MAX_CHARS);
}

#[test]
fn preview_title_falls_back_to_content() {
    let candidate = SearchCandidate {
        record_id: "record".to_string(),
        collection: "docs".to_string(),
        title: "Volcano Needle Title".to_string(),
        content: "body preview text".to_string(),
        metadata: Metadata::new(),
        score: 1.0,
        best_evidence: CandidateEvidence::Title,
    };

    assert_eq!(
        candidate_preview("Volcano Needle Title", &candidate).as_deref(),
        Some("body preview text"),
    );
}

#[test]
fn preview_empty_content_without_evidence_is_none() {
    let candidate = SearchCandidate {
        record_id: "record".to_string(),
        collection: "docs".to_string(),
        title: "Title".to_string(),
        content: String::new(),
        metadata: Metadata::new(),
        score: 1.0,
        best_evidence: CandidateEvidence::None,
    };

    assert_eq!(candidate_preview("Title", &candidate), None);
}

#[test]
fn metadata_filter_json_shape() -> TestResult {
    let filter = MetadataFilter::All(vec![
        MetadataFilter::Equals {
            key: "text".to_string(),
            value: MetadataValue::Text("value".to_string()),
        },
        MetadataFilter::Equals {
            key: "integer".to_string(),
            value: MetadataValue::Integer(7),
        },
        MetadataFilter::Equals {
            key: "float".to_string(),
            value: MetadataValue::Float(1.5),
        },
        MetadataFilter::Equals {
            key: "boolean".to_string(),
            value: MetadataValue::Boolean(true),
        },
    ]);
    let value = serde_json::to_value(&filter)?;

    assert_eq!(
        value,
        serde_json::json!({
            "all": [
                {"equals": {"key": "text", "value": {"text": "value"}}},
                {"equals": {"key": "integer", "value": {"integer": 7}}},
                {"equals": {"key": "float", "value": {"float": 1.5}}},
                {"equals": {"key": "boolean", "value": {"boolean": true}}},
            ]
        }),
    );
    assert_eq!(serde_json::from_value::<MetadataFilter>(value)?, filter);

    Ok(())
}
