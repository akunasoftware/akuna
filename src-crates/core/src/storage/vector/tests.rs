use super::{
    ChunkEntry, RecordEntry, TextSearchQuery, VectorContextOptions,
    VectorDbContext, VectorError, VectorSearchQuery, VectorTextIndex,
    in_memory_context, open_context,
};
use crate::metadata::{Metadata, MetadataFilter, MetadataValue};
use crate::storage::vector::backend::TursoDbContext;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Builds test metadata.
fn metadata(values: &[(&str, MetadataValue)]) -> Metadata {
    values
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect()
}

/// Builds a test record entry.
fn record(
    collection: &str,
    record_id: &str,
    title: &str,
    embedding: [f32; 2],
    content: &str,
    metadata: Metadata,
) -> RecordEntry {
    RecordEntry {
        record_id: record_id.to_string(),
        collection: collection.to_string(),
        title: title.to_string(),
        title_embedding: embedding.to_vec(),
        content: content.to_string(),
        metadata,
    }
}

/// Builds a test chunk entry.
fn chunk(
    collection: &str,
    record_id: &str,
    sequence: u32,
    text: &str,
    embedding: [f32; 2],
    metadata: Metadata,
) -> ChunkEntry {
    ChunkEntry {
        chunk_id: format!("{record_id}:{sequence}"),
        record_id: record_id.to_string(),
        collection: collection.to_string(),
        sequence,
        text: text.to_string(),
        embedding: embedding.to_vec(),
        metadata,
    }
}

/// Builds a test search query.
fn query(
    embedding: [f32; 2],
    collections: &[&str],
    filter: Option<MetadataFilter>,
) -> VectorSearchQuery {
    VectorSearchQuery {
        embedding: embedding.to_vec(),
        collections: collections
            .iter()
            .map(|collection| (*collection).to_string())
            .collect(),
        filter,
        limit: 10,
    }
}

/// Builds full-text test options.
fn fulltext_options() -> VectorContextOptions {
    VectorContextOptions {
        chunk_text_index: true,
        title_text_index: true,
    }
}

/// Builds a test full-text search query.
fn text_query(
    text: &str,
    collections: &[&str],
    filter: Option<MetadataFilter>,
) -> TextSearchQuery {
    TextSearchQuery {
        text: text.to_string(),
        collections: collections
            .iter()
            .map(|collection| (*collection).to_string())
            .collect(),
        filter,
        limit: 10,
    }
}

#[tokio::test]
async fn record_round_trip() -> TestResult {
    let context =
        in_memory_context(2, &VectorContextOptions::default()).await?;
    let first = record(
        "docs",
        "a",
        "Alpha",
        [1.0, 0.0],
        "full alpha content",
        metadata(&[("kind", MetadataValue::Text("note".to_string()))]),
    );
    let second = record(
        "docs",
        "b",
        "Beta",
        [0.0, 1.0],
        "full beta content",
        Metadata::new(),
    );

    context.put_record(&first).await?;
    context.put_record(&second).await?;

    assert_eq!(context.get_record("docs", "a").await?, Some(first.clone()));
    assert_eq!(context.get_record("docs", "missing").await?, None);
    assert_eq!(
        context
            .get_records(&[
                ("docs".to_string(), "b".to_string()),
                ("docs".to_string(), "missing".to_string()),
                ("docs".to_string(), "a".to_string()),
            ])
            .await?,
        vec![second, first],
    );

    Ok(())
}

#[tokio::test]
async fn search_filters_metadata() -> TestResult {
    let context =
        in_memory_context(2, &VectorContextOptions::default()).await?;
    let red = metadata(&[
        ("color", MetadataValue::Text("red".to_string())),
        ("ready", MetadataValue::Boolean(true)),
    ]);
    let blue = metadata(&[
        ("color", MetadataValue::Text("blue".to_string())),
        ("ready", MetadataValue::Boolean(true)),
    ]);
    let record_a =
        record("docs", "a", "Alpha", [1.0, 0.0], "alpha", red.clone());
    let record_b =
        record("docs", "b", "Beta", [0.0, 1.0], "beta", blue.clone());
    let forged = metadata(&[(
        "color\u{1e}text\u{1e}red\u{1f}\u{1f}junk",
        MetadataValue::Text("blue".to_string()),
    )]);
    let record_c =
        record("docs", "c", "Forged", [1.0, 0.0], "forged", forged.clone());

    context.put_record(&record_a).await?;
    context.put_record(&record_b).await?;
    context.put_record(&record_c).await?;
    context
        .put_chunks(
            "docs",
            "a",
            &[chunk("docs", "a", 0, "alpha chunk", [1.0, 0.0], red)],
        )
        .await?;
    context
        .put_chunks(
            "docs",
            "c",
            &[chunk("docs", "c", 0, "forged chunk", [1.0, 0.0], forged)],
        )
        .await?;
    context
        .put_chunks(
            "docs",
            "b",
            &[chunk("docs", "b", 0, "beta chunk", [0.0, 1.0], blue)],
        )
        .await?;

    let equals = MetadataFilter::Equals {
        key: "color".to_string(),
        value: MetadataValue::Text("red".to_string()),
    };
    let all = MetadataFilter::All(vec![
        equals.clone(),
        MetadataFilter::Equals {
            key: "ready".to_string(),
            value: MetadataValue::Boolean(true),
        },
    ]);

    let chunk_results = context
        .search_chunks(&query([1.0, 0.0], &[], Some(equals)))
        .await?;
    assert_eq!(chunk_results.len(), 1);
    assert_eq!(chunk_results[0].record_id, "a");

    let title_results = context
        .search_titles(&query([1.0, 0.0], &[], Some(all.clone())))
        .await?;
    assert_eq!(title_results.len(), 1);
    assert_eq!(title_results[0].record_id, "a");
    assert!(all.matches(&title_results[0].metadata));

    Ok(())
}

#[tokio::test]
async fn search_scopes_collections() -> TestResult {
    let context =
        in_memory_context(2, &VectorContextOptions::default()).await?;
    let docs = record("docs", "a", "Docs", [1.0, 0.0], "docs", Metadata::new());
    let notes =
        record("notes", "a", "Notes", [1.0, 0.0], "notes", Metadata::new());

    context.put_record(&docs).await?;
    context.put_record(&notes).await?;
    context
        .put_chunks(
            "docs",
            "a",
            &[chunk(
                "docs",
                "a",
                0,
                "docs chunk",
                [1.0, 0.0],
                Metadata::new(),
            )],
        )
        .await?;
    context
        .put_chunks(
            "notes",
            "a",
            &[chunk(
                "notes",
                "a",
                0,
                "notes chunk",
                [1.0, 0.0],
                Metadata::new(),
            )],
        )
        .await?;

    assert_eq!(
        context
            .search_chunks(&query([1.0, 0.0], &[], None))
            .await?
            .len(),
        2,
    );
    let scoped = context
        .search_chunks(&query([1.0, 0.0], &["docs"], None))
        .await?;
    assert_eq!(scoped.len(), 1);
    assert_eq!(scoped[0].collection, "docs");

    Ok(())
}

#[tokio::test]
async fn text_search_matches_chunks_and_titles() -> TestResult {
    let options = fulltext_options();
    let context = in_memory_context(2, &options).await?;

    context
        .put_record(&record(
            "docs",
            "a",
            "Rust indexing guide",
            [1.0, 0.0],
            "alpha",
            Metadata::new(),
        ))
        .await?;
    context
        .put_record(&record(
            "docs",
            "b",
            "Cooking notes",
            [0.0, 1.0],
            "beta",
            Metadata::new(),
        ))
        .await?;
    context
        .put_chunks(
            "docs",
            "a",
            &[chunk(
                "docs",
                "a",
                0,
                "lexical retrieval signal",
                [1.0, 0.0],
                Metadata::new(),
            )],
        )
        .await?;
    context
        .put_chunks(
            "docs",
            "b",
            &[chunk(
                "docs",
                "b",
                0,
                "unrelated content",
                [0.0, 1.0],
                Metadata::new(),
            )],
        )
        .await?;

    let chunks = context
        .search_chunks_text(&text_query("lexical", &[], None))
        .await?;
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].record_id, "a");
    assert!(chunks[0].score > 0.0);

    let titles = context
        .search_titles_text(&text_query("indexing", &[], None))
        .await?;
    assert_eq!(titles.len(), 1);
    assert_eq!(titles[0].record_id, "a");
    assert!(titles[0].score > 0.0);

    Ok(())
}

#[tokio::test]
async fn text_search_handles_punctuation_query() -> TestResult {
    let options = fulltext_options();
    let context = in_memory_context(2, &options).await?;

    context
        .put_record(&record(
            "docs",
            "a",
            "Needle++ title",
            [1.0, 0.0],
            "content",
            Metadata::new(),
        ))
        .await?;
    context
        .put_chunks(
            "docs",
            "a",
            &[chunk(
                "docs",
                "a",
                0,
                "needle++ chunk",
                [1.0, 0.0],
                Metadata::new(),
            )],
        )
        .await?;

    let chunks = context
        .search_chunks_text(&text_query("needle++", &[], None))
        .await?;
    assert_eq!(chunks.len(), 1);
    let titles = context
        .search_titles_text(&text_query("needle++", &[], None))
        .await?;
    assert_eq!(titles.len(), 1);
    assert!(
        context
            .search_chunks_text(&text_query("++", &[], None))
            .await?
            .is_empty(),
    );

    context
        .put_chunks(
            "docs",
            "operator",
            &[chunk(
                "docs",
                "operator",
                0,
                "alpha or beta",
                [1.0, 0.0],
                Metadata::new(),
            )],
        )
        .await?;
    assert_eq!(
        context
            .search_chunks_text(&text_query("alpha OR beta", &[], None))
            .await?
            .len(),
        1,
    );

    Ok(())
}

#[tokio::test]
async fn text_search_filters_metadata_and_collections() -> TestResult {
    let options = fulltext_options();
    let context = in_memory_context(2, &options).await?;
    let red = metadata(&[("color", MetadataValue::Text("red".to_string()))]);
    let blue = metadata(&[("color", MetadataValue::Text("blue".to_string()))]);

    for (collection, record_id, metadata) in [
        ("docs", "a", red.clone()),
        ("docs", "b", blue),
        ("notes", "a", red.clone()),
    ] {
        context
            .put_record(&record(
                collection,
                record_id,
                "Shared title",
                [1.0, 0.0],
                "content",
                metadata.clone(),
            ))
            .await?;
        context
            .put_chunks(
                collection,
                record_id,
                &[chunk(
                    collection,
                    record_id,
                    0,
                    "sharedterm body",
                    [1.0, 0.0],
                    metadata,
                )],
            )
            .await?;
    }

    let all = context
        .search_chunks_text(&text_query("sharedterm", &[], None))
        .await?;
    assert_eq!(all.len(), 3);

    let filter = MetadataFilter::Equals {
        key: "color".to_string(),
        value: MetadataValue::Text("red".to_string()),
    };
    let filtered = context
        .search_chunks_text(&text_query("sharedterm", &["docs"], Some(filter)))
        .await?;
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].collection, "docs");
    assert_eq!(filtered[0].record_id, "a");

    let scoped = context
        .search_titles_text(&text_query("shared", &["notes"], None))
        .await?;
    assert_eq!(scoped.len(), 1);
    assert_eq!(scoped[0].collection, "notes");

    Ok(())
}

#[tokio::test]
async fn text_search_requires_enabled_index() -> TestResult {
    let context =
        in_memory_context(2, &VectorContextOptions::default()).await?;

    assert!(matches!(
        context
            .search_chunks_text(&text_query("missing", &[], None))
            .await,
        Err(VectorError::FullTextDisabled {
            target: VectorTextIndex::Chunks,
        }),
    ));
    assert!(matches!(
        context
            .search_titles_text(&text_query("missing", &[], None))
            .await,
        Err(VectorError::FullTextDisabled {
            target: VectorTextIndex::Titles,
        }),
    ));

    Ok(())
}

#[tokio::test]
async fn text_search_persists_after_reopen() -> TestResult {
    let options = fulltext_options();
    let temp_dir = tempfile::tempdir()?;
    {
        let context =
            open_context(temp_dir.path(), 2, &VectorContextOptions::default())
                .await?;
        context
            .put_record(&record(
                "docs",
                "a",
                "Persistent title",
                [1.0, 0.0],
                "content",
                Metadata::new(),
            ))
            .await?;
        context
            .put_chunks(
                "docs",
                "a",
                &[chunk(
                    "docs",
                    "a",
                    0,
                    "persistent chunk",
                    [1.0, 0.0],
                    Metadata::new(),
                )],
            )
            .await?;
    }

    let context = open_context(temp_dir.path(), 2, &options).await?;
    let chunks = context
        .search_chunks_text(&text_query("persistent", &[], None))
        .await?;
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].record_id, "a");
    let titles = context
        .search_titles_text(&text_query("persistent", &[], None))
        .await?;
    assert_eq!(titles.len(), 1);
    assert_eq!(titles[0].record_id, "a");

    drop(context);
    let context = open_context(temp_dir.path(), 2, &options).await?;
    assert_eq!(
        context
            .search_titles_text(&text_query("persistent", &[], None))
            .await?
            .len(),
        1,
    );

    Ok(())
}

#[tokio::test]
async fn record_write_rolls_back_when_fts_write_fails() -> TestResult {
    let options = fulltext_options();
    let context = TursoDbContext::open_in_memory(2, &options).await?;
    context.execute_test_sql("DROP TABLE records_fts;").await?;
    let entry = record(
        "docs",
        "a",
        "Atomic title",
        [1.0, 0.0],
        "content",
        Metadata::new(),
    );

    assert!(context.put_record(&entry).await.is_err());
    assert_eq!(context.get_record("docs", "a").await?, None);

    Ok(())
}

#[tokio::test]
async fn record_and_chunks_write_rolls_back() -> TestResult {
    let options = fulltext_options();
    let context = TursoDbContext::open_in_memory(2, &options).await?;
    let previous = record(
        "docs",
        "a",
        "Previous title",
        [1.0, 0.0],
        "previous content",
        Metadata::new(),
    );
    let previous_chunk = chunk(
        "docs",
        "a",
        0,
        "previous chunk",
        [1.0, 0.0],
        Metadata::new(),
    );
    context
        .put_record_with_chunks(
            &previous,
            std::slice::from_ref(&previous_chunk),
        )
        .await?;
    context.execute_test_sql("DROP TABLE records_fts;").await?;

    let replacement = record(
        "docs",
        "a",
        "Replacement title",
        [0.0, 1.0],
        "replacement content",
        Metadata::new(),
    );
    let replacement_chunk = chunk(
        "docs",
        "a",
        0,
        "replacement chunk",
        [0.0, 1.0],
        Metadata::new(),
    );

    assert!(
        context
            .put_record_with_chunks(&replacement, &[replacement_chunk])
            .await
            .is_err()
    );
    assert_eq!(context.get_record("docs", "a").await?, Some(previous));
    assert_eq!(
        context
            .search_chunks(&query([1.0, 0.0], &[], None))
            .await?
            .into_iter()
            .map(|result| result.text)
            .collect::<Vec<_>>(),
        vec![previous_chunk.text],
    );

    Ok(())
}

#[tokio::test]
async fn delete_rolls_back_when_fts_write_fails() -> TestResult {
    let options = fulltext_options();
    let context = TursoDbContext::open_in_memory(2, &options).await?;
    let entry = record(
        "docs",
        "a",
        "Atomic title",
        [1.0, 0.0],
        "content",
        Metadata::new(),
    );
    let entry_chunk =
        chunk("docs", "a", 0, "atomic chunk", [1.0, 0.0], Metadata::new());
    context
        .put_record_with_chunks(&entry, std::slice::from_ref(&entry_chunk))
        .await?;
    context.execute_test_sql("DROP TABLE records_fts;").await?;

    assert!(context.delete_record("docs", "a").await.is_err());
    assert_eq!(context.get_record("docs", "a").await?, Some(entry));
    assert_eq!(
        context
            .search_chunks(&query([1.0, 0.0], &[], None))
            .await?
            .into_iter()
            .map(|result| result.text)
            .collect::<Vec<_>>(),
        vec![entry_chunk.text],
    );

    Ok(())
}

#[tokio::test]
async fn fulltext_tracks_writes_while_disabled_after_reopen() -> TestResult {
    let temp_dir = tempfile::tempdir()?;
    let options = fulltext_options();
    let old = record(
        "docs",
        "a",
        "Old title",
        [1.0, 0.0],
        "old content",
        Metadata::new(),
    );
    let old_chunk =
        chunk("docs", "a", 0, "old chunk", [1.0, 0.0], Metadata::new());
    {
        let context = open_context(temp_dir.path(), 2, &options).await?;
        context
            .put_record_with_chunks(&old, std::slice::from_ref(&old_chunk))
            .await?;
    }

    let replacement = record(
        "docs",
        "a",
        "Replacement title",
        [0.0, 1.0],
        "replacement content",
        Metadata::new(),
    );
    let replacement_chunk = chunk(
        "docs",
        "a",
        0,
        "replacement chunk",
        [0.0, 1.0],
        Metadata::new(),
    );
    {
        let context =
            open_context(temp_dir.path(), 2, &VectorContextOptions::default())
                .await?;
        context
            .put_record_with_chunks(&replacement, &[replacement_chunk])
            .await?;
    }

    {
        let context = open_context(temp_dir.path(), 2, &options).await?;
        assert!(
            context
                .search_chunks_text(&text_query("old", &[], None))
                .await?
                .is_empty()
        );
        assert!(
            context
                .search_titles_text(&text_query("old", &[], None))
                .await?
                .is_empty()
        );
        assert_eq!(
            context
                .search_chunks_text(&text_query("replacement", &[], None))
                .await?
                .len(),
            1,
        );
    }
    {
        let context =
            open_context(temp_dir.path(), 2, &VectorContextOptions::default())
                .await?;
        context.delete_record("docs", "a").await?;
    }

    let context = open_context(temp_dir.path(), 2, &options).await?;
    assert!(
        context
            .search_chunks_text(&text_query("replacement", &[], None))
            .await?
            .is_empty()
    );
    assert!(
        context
            .search_titles_text(&text_query("replacement", &[], None))
            .await?
            .is_empty()
    );

    Ok(())
}

#[tokio::test]
async fn text_search_finds_writes_after_open() -> TestResult {
    let options = fulltext_options();
    let context = in_memory_context(2, &options).await?;

    context
        .put_record(&record(
            "docs",
            "seed",
            "Seed title",
            [1.0, 0.0],
            "content",
            Metadata::new(),
        ))
        .await?;
    context
        .put_chunks(
            "docs",
            "seed",
            &[chunk(
                "docs",
                "seed",
                0,
                "seed chunk",
                [1.0, 0.0],
                Metadata::new(),
            )],
        )
        .await?;
    assert_eq!(
        context
            .search_chunks_text(&text_query("seed", &[], None))
            .await?
            .len(),
        1,
    );

    context
        .put_record(&record(
            "docs",
            "later",
            "Later needle",
            [0.0, 1.0],
            "content",
            Metadata::new(),
        ))
        .await?;
    context
        .put_chunks(
            "docs",
            "later",
            &[chunk(
                "docs",
                "later",
                0,
                "later needle chunk",
                [0.0, 1.0],
                Metadata::new(),
            )],
        )
        .await?;

    let chunks = context
        .search_chunks_text(&text_query("needle", &[], None))
        .await?;
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].record_id, "later");
    let titles = context
        .search_titles_text(&text_query("needle", &[], None))
        .await?;
    assert_eq!(titles.len(), 1);
    assert_eq!(titles[0].record_id, "later");

    Ok(())
}

#[tokio::test]
async fn chunks_replace_and_clear() -> TestResult {
    let context =
        in_memory_context(2, &VectorContextOptions::default()).await?;
    context
        .put_chunks(
            "docs",
            "a",
            &[
                chunk("docs", "a", 0, "old one", [1.0, 0.0], Metadata::new()),
                chunk("docs", "a", 1, "old two", [0.0, 1.0], Metadata::new()),
            ],
        )
        .await?;
    assert!(
        context
            .put_chunks(
                "docs",
                "a",
                &[
                    chunk(
                        "docs",
                        "a",
                        2,
                        "bad one",
                        [1.0, 0.0],
                        Metadata::new()
                    ),
                    chunk(
                        "docs",
                        "a",
                        2,
                        "bad two",
                        [0.0, 1.0],
                        Metadata::new()
                    ),
                ],
            )
            .await
            .is_err(),
    );
    let mut old_texts = context
        .search_chunks(&query([1.0, 0.0], &["docs"], None))
        .await?
        .into_iter()
        .map(|result| result.text)
        .collect::<Vec<_>>();
    old_texts.sort();
    assert_eq!(old_texts, vec!["old one", "old two"]);

    context
        .put_chunks(
            "docs",
            "a",
            &[chunk("docs", "a", 2, "new", [0.0, 1.0], Metadata::new())],
        )
        .await?;
    let replaced = context
        .search_chunks(&query([0.0, 1.0], &["docs"], None))
        .await?;
    assert_eq!(replaced.len(), 1);
    assert_eq!(replaced[0].text, "new");

    context.put_chunks("docs", "a", &[]).await?;
    assert!(
        context
            .search_chunks(&query([0.0, 1.0], &["docs"], None))
            .await?
            .is_empty(),
    );

    Ok(())
}

#[tokio::test]
async fn delete_is_idempotent() -> TestResult {
    let context =
        in_memory_context(2, &VectorContextOptions::default()).await?;
    let entry =
        record("docs", "a", "Alpha", [1.0, 0.0], "alpha", Metadata::new());
    context.put_record(&entry).await?;
    context
        .put_chunks(
            "docs",
            "a",
            &[chunk("docs", "a", 0, "alpha", [1.0, 0.0], Metadata::new())],
        )
        .await?;

    context.delete_record("docs", "a").await?;
    context.delete_record("docs", "a").await?;

    assert_eq!(context.get_record("docs", "a").await?, None);
    assert!(
        context
            .search_chunks(&query([1.0, 0.0], &[], None))
            .await?
            .is_empty(),
    );
    assert!(
        context
            .search_titles(&query([1.0, 0.0], &[], None))
            .await?
            .is_empty(),
    );

    Ok(())
}

#[tokio::test]
async fn dimension_mismatch_errors() -> TestResult {
    let context =
        in_memory_context(2, &VectorContextOptions::default()).await?;
    let mut bad_record =
        record("docs", "a", "Alpha", [1.0, 0.0], "alpha", Metadata::new());
    bad_record.title_embedding = vec![1.0];
    assert!(matches!(
        context.put_record(&bad_record).await,
        Err(VectorError::DimensionMismatch {
            expected: 2,
            actual: 1
        }),
    ));

    let bad_chunk = ChunkEntry {
        embedding: vec![1.0, 0.0, 0.0],
        ..chunk("docs", "a", 0, "bad", [1.0, 0.0], Metadata::new())
    };
    assert!(matches!(
        context.put_chunks("docs", "a", &[bad_chunk]).await,
        Err(VectorError::DimensionMismatch {
            expected: 2,
            actual: 3
        }),
    ));
    assert!(matches!(
        context
            .search_chunks(&VectorSearchQuery {
                embedding: vec![1.0],
                collections: Vec::new(),
                filter: None,
                limit: 1,
            })
            .await,
        Err(VectorError::DimensionMismatch {
            expected: 2,
            actual: 1
        }),
    ));
    assert!(matches!(
        context
            .put_chunks(
                "docs",
                "a",
                &[chunk("other", "a", 0, "bad", [1.0, 0.0], Metadata::new())],
            )
            .await,
        Err(VectorError::InvalidChunkEntry { .. }),
    ));

    let temp_dir = tempfile::tempdir()?;
    let persistent =
        open_context(temp_dir.path(), 2, &VectorContextOptions::default())
            .await?;
    drop(persistent);
    let result =
        open_context(temp_dir.path(), 3, &VectorContextOptions::default())
            .await;
    assert!(matches!(
        result,
        Err(VectorError::DimensionMismatch {
            expected: 3,
            actual: 2
        }),
    ));

    Ok(())
}

#[tokio::test]
async fn memory_lifecycle_removes_path() -> TestResult {
    let path = {
        let context =
            TursoDbContext::open_in_memory(2, &VectorContextOptions::default())
                .await?;
        let Some(path) = context.temp_path() else {
            panic!("in-memory context should expose temp path in tests");
        };
        let path = path.to_path_buf();
        assert!(path.exists());
        path
    };

    assert!(!path.exists());

    Ok(())
}
