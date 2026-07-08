use std::fs;
use std::path::PathBuf;

use async_trait::async_trait;
use libsql::{Builder, Connection, Database, Row, Value};
use tempfile::TempDir;

use crate::metadata::{Metadata, MetadataFilter, MetadataValue};
use crate::storage::vector::{
    ChunkEntry, ChunkSearchResult, RecordEntry, RecordSearchResult,
    TextSearchQuery, VectorContextOptions, VectorDbContext, VectorError,
    VectorSearchQuery, VectorTarget, VectorTextIndex, VectorWriteOperation,
};

const ENGINE_NAME: &str = "libsql";
const DB_FILE: &str = "vector.db";
const DIMENSIONS_KEY: &str = "dimensions";
const CHUNKS_VECTOR_INDEX: &str = "chunks_embedding_idx";
const RECORDS_VECTOR_INDEX: &str = "records_title_embedding_idx";

/// Turso-backed vector storage context.
pub(crate) struct TursoDbContext {
    dimensions: usize,
    options: VectorContextOptions,
    chunk_text_index: bool,
    title_text_index: bool,
    _database: Database,
    connection: Connection,
    #[cfg(test)]
    temp_path: Option<PathBuf>,
    _temp_dir: Option<TempDir>,
}

impl TursoDbContext {
    /// Opens a Turso-backed vector context rooted at the given path.
    pub async fn open(
        persist_at: PathBuf,
        dimensions: usize,
        options: &VectorContextOptions,
    ) -> Result<Self, VectorError> {
        Self::open_inner(persist_at, dimensions, options.clone(), None).await
    }

    /// Opens an ephemeral Turso-backed vector context.
    pub async fn open_in_memory(
        dimensions: usize,
        options: &VectorContextOptions,
    ) -> Result<Self, VectorError> {
        let temp_dir =
            tempfile::tempdir().map_err(|source| VectorError::DbInit {
                engine: ENGINE_NAME,
                source: Box::new(source),
            })?;
        let path = temp_dir.path().to_path_buf();

        Self::open_inner(path, dimensions, options.clone(), Some(temp_dir))
            .await
    }

    #[cfg(test)]
    pub(crate) fn temp_path(&self) -> Option<&std::path::Path> {
        self.temp_path.as_deref()
    }

    #[cfg(test)]
    pub(crate) async fn execute_test_sql(
        &self,
        sql: &str,
    ) -> Result<(), VectorError> {
        self.connection
            .execute_batch(sql)
            .await
            .map(|_| ())
            .map_err(db_init_error)
    }

    async fn open_inner(
        path: PathBuf,
        dimensions: usize,
        options: VectorContextOptions,
        temp_dir: Option<TempDir>,
    ) -> Result<Self, VectorError> {
        checked_dimensions(dimensions)?;
        fs::create_dir_all(&path).map_err(|source| VectorError::DbInit {
            engine: ENGINE_NAME,
            source: Box::new(source),
        })?;
        let db_path = path.join(DB_FILE);
        let database = Builder::new_local(&db_path)
            .build()
            .await
            .map_err(db_init_error)?;
        let connection = database.connect().map_err(db_init_error)?;
        let mut context = Self {
            dimensions,
            options,
            chunk_text_index: false,
            title_text_index: false,
            _database: database,
            connection,
            #[cfg(test)]
            temp_path: temp_dir.as_ref().map(|dir| dir.path().to_path_buf()),
            _temp_dir: temp_dir,
        };

        (context.chunk_text_index, context.title_text_index) =
            context.initialise().await?;
        Ok(context)
    }

    async fn initialise(&self) -> Result<(bool, bool), VectorError> {
        self.connection
            .execute_batch(&format!(
                r#"
                CREATE TABLE IF NOT EXISTS vector_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS records (
                    collection TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    title_embedding F32_BLOB({dimensions}) NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    metadata_tokens TEXT NOT NULL,
                    PRIMARY KEY (collection, record_id)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    collection TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding F32_BLOB({dimensions}) NOT NULL,
                    metadata_json TEXT NOT NULL,
                    metadata_tokens TEXT NOT NULL,
                    PRIMARY KEY (collection, record_id, chunk_id)
                );
                CREATE INDEX IF NOT EXISTS records_collection_idx
                    ON records(collection);
                CREATE INDEX IF NOT EXISTS chunks_record_idx
                    ON chunks(collection, record_id);
                CREATE INDEX IF NOT EXISTS chunks_collection_idx
                    ON chunks(collection);
                CREATE INDEX IF NOT EXISTS {CHUNKS_VECTOR_INDEX}
                    ON chunks(libsql_vector_idx(embedding));
                CREATE INDEX IF NOT EXISTS {RECORDS_VECTOR_INDEX}
                    ON records(libsql_vector_idx(title_embedding));
                "#,
                dimensions = self.dimensions,
            ))
            .await
            .map_err(db_init_error)?;
        self.ensure_dimensions().await?;
        let chunk_text_index = if self.options.chunk_text_index {
            self.ensure_chunk_fts().await?;
            true
        } else {
            self.table_exists("chunks_fts").await?
        };
        let title_text_index = if self.options.title_text_index {
            self.ensure_title_fts().await?;
            true
        } else {
            self.table_exists("records_fts").await?
        };

        Ok((chunk_text_index, title_text_index))
    }

    async fn ensure_dimensions(&self) -> Result<(), VectorError> {
        let mut rows = self
            .connection
            .query(
                "SELECT value FROM vector_meta WHERE key = ?1",
                [DIMENSIONS_KEY],
            )
            .await
            .map_err(query_error)?;
        let stored = rows.next().await.map_err(query_error)?;
        if let Some(row) = stored {
            let actual = text_value(&row, 0)?
                .parse::<usize>()
                .map_err(deserialization_error)?;
            if actual == self.dimensions {
                return Ok(());
            }

            return Err(VectorError::DimensionMismatch {
                expected: self.dimensions,
                actual,
            });
        }

        self.connection
            .execute(
                "INSERT INTO vector_meta (key, value) VALUES (?1, ?2)",
                (DIMENSIONS_KEY, self.dimensions.to_string()),
            )
            .await
            .map_err(|source| {
                write_error(source, VectorWriteOperation::Put, meta_target())
            })?;

        Ok(())
    }

    async fn ensure_chunk_fts(&self) -> Result<(), VectorError> {
        if self.table_exists("chunks_fts").await? {
            return Ok(());
        }

        let tx = self.connection.transaction().await.map_err(db_init_error)?;
        tx.execute_batch(
                r#"
                CREATE VIRTUAL TABLE chunks_fts
                    USING fts5(collection UNINDEXED, record_id UNINDEXED, chunk_id UNINDEXED, text);
                INSERT INTO chunks_fts (collection, record_id, chunk_id, text)
                    SELECT collection, record_id, chunk_id, text FROM chunks;
                "#,
        )
        .await
        .map_err(db_init_error)?;
        tx.commit().await.map_err(db_init_error)?;

        Ok(())
    }

    async fn ensure_title_fts(&self) -> Result<(), VectorError> {
        if self.table_exists("records_fts").await? {
            return Ok(());
        }

        let tx = self.connection.transaction().await.map_err(db_init_error)?;
        tx.execute_batch(
                r#"
                CREATE VIRTUAL TABLE records_fts
                    USING fts5(collection UNINDEXED, record_id UNINDEXED, title);
                INSERT INTO records_fts (collection, record_id, title)
                    SELECT collection, record_id, title FROM records;
                "#,
        )
        .await
        .map_err(db_init_error)?;
        tx.commit().await.map_err(db_init_error)?;

        Ok(())
    }

    /// Returns whether a SQLite table already exists.
    async fn table_exists(&self, table: &str) -> Result<bool, VectorError> {
        let mut rows = self
            .connection
            .query(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1",
                [table],
            )
            .await
            .map_err(db_init_error)?;

        Ok(rows.next().await.map_err(db_init_error)?.is_some())
    }

    fn validate_embedding(&self, actual: usize) -> Result<(), VectorError> {
        if actual == self.dimensions {
            return Ok(());
        }

        Err(VectorError::DimensionMismatch {
            expected: self.dimensions,
            actual,
        })
    }

    fn validate_chunks(
        &self,
        collection: &str,
        record_id: &str,
        chunks: &[ChunkEntry],
    ) -> Result<(), VectorError> {
        for chunk in chunks {
            if chunk.collection != collection || chunk.record_id != record_id {
                return Err(VectorError::InvalidChunkEntry {
                    expected_collection: collection.to_string(),
                    expected_record_id: record_id.to_string(),
                    chunk_id: chunk.chunk_id.clone(),
                    actual_collection: chunk.collection.clone(),
                    actual_record_id: chunk.record_id.clone(),
                });
            }
            self.validate_embedding(chunk.embedding.len())?;
        }

        Ok(())
    }

    async fn query_map<T>(
        &self,
        sql: String,
        params: Vec<Value>,
        mut map: impl FnMut(&Row) -> Result<T, VectorError>,
    ) -> Result<Vec<T>, VectorError> {
        let mut rows = self
            .connection
            .query(&sql, params)
            .await
            .map_err(query_error)?;
        let mut values = Vec::new();
        while let Some(row) = rows.next().await.map_err(query_error)? {
            values.push(map(&row)?);
        }

        Ok(values)
    }
}

#[async_trait]
impl VectorDbContext for TursoDbContext {
    async fn put_record_with_chunks(
        &self,
        record: &RecordEntry,
        chunks: &[ChunkEntry],
    ) -> Result<(), VectorError> {
        self.validate_embedding(record.title_embedding.len())?;
        self.validate_chunks(&record.collection, &record.record_id, chunks)?;
        let record_metadata_json = metadata_json(&record.metadata)?;
        let record_metadata_tokens = metadata_tokens(&record.metadata);
        let title_embedding = embedding_json(&record.title_embedding)?;
        let chunks = chunks
            .iter()
            .map(|chunk| {
                Ok((
                    chunk,
                    metadata_json(&chunk.metadata)?,
                    metadata_tokens(&chunk.metadata),
                    embedding_json(&chunk.embedding)?,
                ))
            })
            .collect::<Result<Vec<_>, VectorError>>()?;
        let target = record_target(&record.collection, &record.record_id);
        let chunks_target =
            chunks_target(&record.collection, &record.record_id);
        let tx = self.connection.transaction().await.map_err(|source| {
            write_error(source, VectorWriteOperation::Put, target.clone())
        })?;

        if self.chunk_text_index {
            tx.execute(
                "DELETE FROM chunks_fts WHERE collection = ?1 AND record_id = ?2",
                (record.collection.as_str(), record.record_id.as_str()),
            )
            .await
            .map_err(|source| {
                write_error(
                    source,
                    VectorWriteOperation::Delete,
                    chunks_target.clone(),
                )
            })?;
        }
        tx.execute(
            "DELETE FROM chunks WHERE collection = ?1 AND record_id = ?2",
            (record.collection.as_str(), record.record_id.as_str()),
        )
        .await
        .map_err(|source| {
            write_error(
                source,
                VectorWriteOperation::Delete,
                chunks_target.clone(),
            )
        })?;
        if self.title_text_index {
            tx.execute(
                "DELETE FROM records_fts WHERE collection = ?1 AND record_id = ?2",
                (record.collection.as_str(), record.record_id.as_str()),
            )
            .await
            .map_err(|source| {
                write_error(source, VectorWriteOperation::Delete, target.clone())
            })?;
        }
        tx.execute(
            "DELETE FROM records WHERE collection = ?1 AND record_id = ?2",
            (record.collection.as_str(), record.record_id.as_str()),
        )
        .await
        .map_err(|source| {
            write_error(source, VectorWriteOperation::Delete, target.clone())
        })?;

        tx.execute(
            r#"
            INSERT INTO records
                (collection, record_id, title, title_embedding, content, metadata_json, metadata_tokens)
            VALUES (?1, ?2, ?3, vector(?4), ?5, ?6, ?7)
            "#,
            (
                record.collection.clone(),
                record.record_id.clone(),
                record.title.clone(),
                title_embedding,
                record.content.clone(),
                record_metadata_json,
                record_metadata_tokens,
            ),
        )
        .await
        .map_err(|source| {
            write_error(source, VectorWriteOperation::Put, target.clone())
        })?;
        if self.title_text_index {
            tx.execute(
                r#"
                INSERT INTO records_fts (collection, record_id, title)
                VALUES (?1, ?2, ?3)
                "#,
                (
                    record.collection.clone(),
                    record.record_id.clone(),
                    record.title.clone(),
                ),
            )
            .await
            .map_err(|source| {
                write_error(source, VectorWriteOperation::Put, target.clone())
            })?;
        }
        for (chunk, metadata_json, metadata_tokens, embedding) in chunks {
            tx.execute(
                r#"
                INSERT INTO chunks
                    (collection, record_id, chunk_id, sequence, text, embedding, metadata_json, metadata_tokens)
                VALUES (?1, ?2, ?3, ?4, ?5, vector(?6), ?7, ?8)
                "#,
                (
                    chunk.collection.clone(),
                    chunk.record_id.clone(),
                    chunk.chunk_id.clone(),
                    i64::from(chunk.sequence),
                    chunk.text.clone(),
                    embedding,
                    metadata_json,
                    metadata_tokens,
                ),
            )
            .await
            .map_err(|source| {
                write_error(source, VectorWriteOperation::Put, chunks_target.clone())
            })?;
            if self.chunk_text_index {
                tx.execute(
                    r#"
                    INSERT INTO chunks_fts (collection, record_id, chunk_id, text)
                    VALUES (?1, ?2, ?3, ?4)
                    "#,
                    (
                        chunk.collection.clone(),
                        chunk.record_id.clone(),
                        chunk.chunk_id.clone(),
                        chunk.text.clone(),
                    ),
                )
                .await
                .map_err(|source| {
                    write_error(source, VectorWriteOperation::Put, chunks_target.clone())
                })?;
            }
        }
        tx.commit().await.map_err(|source| {
            write_error(source, VectorWriteOperation::Put, target)
        })?;

        Ok(())
    }

    async fn put_chunks(
        &self,
        collection: &str,
        record_id: &str,
        chunks: &[ChunkEntry],
    ) -> Result<(), VectorError> {
        self.validate_chunks(collection, record_id, chunks)?;
        let tx = self.connection.transaction().await.map_err(|source| {
            write_error(
                source,
                VectorWriteOperation::Put,
                chunks_target(collection, record_id),
            )
        })?;
        if self.chunk_text_index {
            tx.execute(
                "DELETE FROM chunks_fts WHERE collection = ?1 AND record_id = ?2",
                (collection, record_id),
            )
            .await
            .map_err(|source| {
                write_error(
                    source,
                    VectorWriteOperation::Delete,
                    chunks_target(collection, record_id),
                )
            })?;
        }
        tx.execute(
            "DELETE FROM chunks WHERE collection = ?1 AND record_id = ?2",
            (collection, record_id),
        )
        .await
        .map_err(|source| {
            write_error(
                source,
                VectorWriteOperation::Delete,
                chunks_target(collection, record_id),
            )
        })?;

        for chunk in chunks {
            let metadata_json = metadata_json(&chunk.metadata)?;
            let metadata_tokens = metadata_tokens(&chunk.metadata);
            let embedding = embedding_json(&chunk.embedding)?;
            tx
                .execute(
                    r#"
                    INSERT INTO chunks
                        (collection, record_id, chunk_id, sequence, text, embedding, metadata_json, metadata_tokens)
                    VALUES (?1, ?2, ?3, ?4, ?5, vector(?6), ?7, ?8)
                    "#,
                    (
                        chunk.collection.clone(),
                        chunk.record_id.clone(),
                        chunk.chunk_id.clone(),
                        i64::from(chunk.sequence),
                        chunk.text.clone(),
                        embedding,
                        metadata_json,
                        metadata_tokens,
                    ),
                )
                .await
                .map_err(|source| {
                    write_error(source, VectorWriteOperation::Put, chunks_target(collection, record_id))
                })?;
            if self.chunk_text_index {
                tx
                    .execute(
                        r#"
                        INSERT INTO chunks_fts (collection, record_id, chunk_id, text)
                        VALUES (?1, ?2, ?3, ?4)
                        "#,
                        (
                            chunk.collection.clone(),
                            chunk.record_id.clone(),
                            chunk.chunk_id.clone(),
                            chunk.text.clone(),
                        ),
                    )
                    .await
                    .map_err(|source| {
                        write_error(
                            source,
                            VectorWriteOperation::Put,
                            chunks_target(collection, record_id),
                        )
                    })?;
            }
        }
        tx.commit().await.map_err(|source| {
            write_error(
                source,
                VectorWriteOperation::Put,
                chunks_target(collection, record_id),
            )
        })?;

        Ok(())
    }

    async fn put_record(
        &self,
        record: &RecordEntry,
    ) -> Result<(), VectorError> {
        self.validate_embedding(record.title_embedding.len())?;
        let metadata_json = metadata_json(&record.metadata)?;
        let metadata_tokens = metadata_tokens(&record.metadata);
        let title_embedding = embedding_json(&record.title_embedding)?;
        let tx = self.connection.transaction().await.map_err(|source| {
            write_error(
                source,
                VectorWriteOperation::Put,
                record_target(&record.collection, &record.record_id),
            )
        })?;

        tx
            .execute(
                r#"
                INSERT OR REPLACE INTO records
                    (collection, record_id, title, title_embedding, content, metadata_json, metadata_tokens)
                VALUES (?1, ?2, ?3, vector(?4), ?5, ?6, ?7)
                "#,
                (
                    record.collection.clone(),
                    record.record_id.clone(),
                    record.title.clone(),
                    title_embedding,
                    record.content.clone(),
                    metadata_json,
                    metadata_tokens,
                ),
            )
            .await
            .map_err(|source| {
                write_error(source, VectorWriteOperation::Put, record_target(&record.collection, &record.record_id))
        })?;
        if self.title_text_index {
            tx
                .execute(
                    "DELETE FROM records_fts WHERE collection = ?1 AND record_id = ?2",
                    (record.collection.as_str(), record.record_id.as_str()),
                )
                .await
                .map_err(|source| {
                    write_error(
                        source,
                        VectorWriteOperation::Put,
                        record_target(&record.collection, &record.record_id),
                    )
                })?;
            tx.execute(
                r#"
                    INSERT INTO records_fts (collection, record_id, title)
                    VALUES (?1, ?2, ?3)
                    "#,
                (
                    record.collection.clone(),
                    record.record_id.clone(),
                    record.title.clone(),
                ),
            )
            .await
            .map_err(|source| {
                write_error(
                    source,
                    VectorWriteOperation::Put,
                    record_target(&record.collection, &record.record_id),
                )
            })?;
        }
        tx.commit().await.map_err(|source| {
            write_error(
                source,
                VectorWriteOperation::Put,
                record_target(&record.collection, &record.record_id),
            )
        })?;

        Ok(())
    }

    async fn delete_record(
        &self,
        collection: &str,
        record_id: &str,
    ) -> Result<(), VectorError> {
        let chunks_target = chunks_target(collection, record_id);
        let record_target = record_target(collection, record_id);
        let tx = self.connection.transaction().await.map_err(|source| {
            write_error(
                source,
                VectorWriteOperation::Delete,
                record_target.clone(),
            )
        })?;

        if self.chunk_text_index {
            tx.execute(
                "DELETE FROM chunks_fts WHERE collection = ?1 AND record_id = ?2",
                (collection, record_id),
            )
            .await
            .map_err(|source| {
                write_error(
                    source,
                    VectorWriteOperation::Delete,
                    chunks_target.clone(),
                )
            })?;
        }
        tx.execute(
            "DELETE FROM chunks WHERE collection = ?1 AND record_id = ?2",
            (collection, record_id),
        )
        .await
        .map_err(|source| {
            write_error(source, VectorWriteOperation::Delete, chunks_target)
        })?;
        if self.title_text_index {
            tx.execute(
                "DELETE FROM records_fts WHERE collection = ?1 AND record_id = ?2",
                (collection, record_id),
            )
            .await
            .map_err(|source| {
                write_error(
                    source,
                    VectorWriteOperation::Delete,
                    record_target.clone(),
                )
            })?;
        }
        tx.execute(
            "DELETE FROM records WHERE collection = ?1 AND record_id = ?2",
            (collection, record_id),
        )
        .await
        .map_err(|source| {
            write_error(
                source,
                VectorWriteOperation::Delete,
                record_target.clone(),
            )
        })?;
        tx.commit().await.map_err(|source| {
            write_error(source, VectorWriteOperation::Delete, record_target)
        })
    }

    async fn get_record(
        &self,
        collection: &str,
        record_id: &str,
    ) -> Result<Option<RecordEntry>, VectorError> {
        let mut rows = self
            .connection
            .query(
                r#"
                SELECT record_id, collection, title, vector_extract(title_embedding), content, metadata_json
                FROM records
                WHERE collection = ?1 AND record_id = ?2
                "#,
                (collection, record_id),
            )
            .await
            .map_err(query_error)?;
        let Some(row) = rows.next().await.map_err(query_error)? else {
            return Ok(None);
        };

        row_to_record(&row).map(Some)
    }

    async fn get_records(
        &self,
        keys: &[(String, String)],
    ) -> Result<Vec<RecordEntry>, VectorError> {
        let mut records = Vec::new();
        for (collection, record_id) in keys {
            if let Some(record) = self.get_record(collection, record_id).await?
            {
                records.push(record);
            }
        }

        Ok(records)
    }

    async fn search_chunks(
        &self,
        query: &VectorSearchQuery,
    ) -> Result<Vec<ChunkSearchResult>, VectorError> {
        self.validate_embedding(query.embedding.len())?;
        if query.limit == 0 {
            return Ok(Vec::new());
        }
        let embedding = embedding_json(&query.embedding)?;

        if query.collections.is_empty() && query.filter.is_none() {
            return self
                .query_map(
                    format!(
                        r#"
                        SELECT c.chunk_id, c.record_id, c.collection, c.text, c.metadata_json,
                            1.0 - vector_distance_cos(c.embedding, vector(?1))
                        FROM vector_top_k('{CHUNKS_VECTOR_INDEX}', ?1, ?2) AS v
                        JOIN chunks c ON c.rowid = v.id
                        ORDER BY vector_distance_cos(c.embedding, vector(?3)) ASC,
                            c.collection, c.record_id, c.chunk_id
                        "#,
                    ),
                    vec![
                        Value::Text(embedding.clone()),
                        Value::Integer(limit_i64(query.limit)?),
                        Value::Text(embedding),
                    ],
                    chunk_score_result,
                )
                .await;
        }

        let (where_clause, params) =
            sql_filter("c", &query.collections, query.filter.as_ref());
        let mut all_params = vec![Value::Text(embedding.clone())];
        all_params.extend(params);
        all_params.push(Value::Text(embedding));
        all_params.push(Value::Integer(limit_i64(query.limit)?));
        self
            .query_map(
                format!(
                    r#"
                    SELECT c.chunk_id, c.record_id, c.collection, c.text, c.metadata_json,
                        1.0 - vector_distance_cos(c.embedding, vector(?))
                    FROM chunks c{where_clause}
                    ORDER BY vector_distance_cos(c.embedding, vector(?)) ASC,
                        c.collection, c.record_id, c.chunk_id
                    LIMIT ?
                    "#,
                ),
                all_params,
                chunk_score_result,
            )
            .await
    }

    async fn search_titles(
        &self,
        query: &VectorSearchQuery,
    ) -> Result<Vec<RecordSearchResult>, VectorError> {
        self.validate_embedding(query.embedding.len())?;
        if query.limit == 0 {
            return Ok(Vec::new());
        }
        let embedding = embedding_json(&query.embedding)?;

        if query.collections.is_empty() && query.filter.is_none() {
            return self
                .query_map(
                    format!(
                        r#"
                        SELECT r.record_id, r.collection, r.title, r.metadata_json,
                            1.0 - vector_distance_cos(r.title_embedding, vector(?1))
                        FROM vector_top_k('{RECORDS_VECTOR_INDEX}', ?1, ?2) AS v
                        JOIN records r ON r.rowid = v.id
                        ORDER BY vector_distance_cos(r.title_embedding, vector(?3)) ASC,
                            r.collection, r.record_id
                        "#,
                    ),
                    vec![
                        Value::Text(embedding.clone()),
                        Value::Integer(limit_i64(query.limit)?),
                        Value::Text(embedding),
                    ],
                    title_score_result,
                )
                .await;
        }

        let (where_clause, params) =
            sql_filter("r", &query.collections, query.filter.as_ref());
        let mut all_params = vec![Value::Text(embedding.clone())];
        all_params.extend(params);
        all_params.push(Value::Text(embedding));
        all_params.push(Value::Integer(limit_i64(query.limit)?));
        self
            .query_map(
                format!(
                    r#"
                    SELECT r.record_id, r.collection, r.title, r.metadata_json,
                        1.0 - vector_distance_cos(r.title_embedding, vector(?))
                    FROM records r{where_clause}
                    ORDER BY vector_distance_cos(r.title_embedding, vector(?)) ASC,
                        r.collection, r.record_id
                    LIMIT ?
                    "#,
                ),
                all_params,
                title_score_result,
            )
            .await
    }

    async fn search_chunks_text(
        &self,
        query: &TextSearchQuery,
    ) -> Result<Vec<ChunkSearchResult>, VectorError> {
        if !self.options.chunk_text_index {
            return Err(VectorError::FullTextDisabled {
                target: VectorTextIndex::Chunks,
            });
        }
        if query.limit == 0 || query.text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let text = fts_query(&query.text);
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let (filter_clause, mut params) =
            sql_filter("c", &query.collections, query.filter.as_ref());
        let mut all_params = vec![Value::Text(text)];
        all_params.append(&mut params);
        all_params.push(Value::Integer(limit_i64(query.limit)?));
        let where_tail = filter_clause.replacen(" WHERE ", " AND ", 1);
        self
            .query_map(
                format!(
                    r#"
                    SELECT c.chunk_id, c.record_id, c.collection, c.text, c.metadata_json, -bm25(chunks_fts)
                    FROM chunks_fts
                    JOIN chunks c
                        ON c.collection = chunks_fts.collection
                        AND c.record_id = chunks_fts.record_id
                        AND c.chunk_id = chunks_fts.chunk_id
                    WHERE chunks_fts MATCH ?1{where_tail}
                    ORDER BY bm25(chunks_fts), c.collection, c.record_id, c.chunk_id
                    LIMIT ?{}
                    "#,
                    all_params.len(),
                ),
                all_params,
                chunk_score_result,
            )
            .await
    }

    async fn search_titles_text(
        &self,
        query: &TextSearchQuery,
    ) -> Result<Vec<RecordSearchResult>, VectorError> {
        if !self.options.title_text_index {
            return Err(VectorError::FullTextDisabled {
                target: VectorTextIndex::Titles,
            });
        }
        if query.limit == 0 || query.text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let text = fts_query(&query.text);
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let (filter_clause, mut params) =
            sql_filter("r", &query.collections, query.filter.as_ref());
        let mut all_params = vec![Value::Text(text)];
        all_params.append(&mut params);
        all_params.push(Value::Integer(limit_i64(query.limit)?));
        let where_tail = filter_clause.replacen(" WHERE ", " AND ", 1);
        self
            .query_map(
                format!(
                    r#"
                    SELECT r.record_id, r.collection, r.title, r.metadata_json, -bm25(records_fts)
                    FROM records_fts
                    JOIN records r
                        ON r.collection = records_fts.collection
                        AND r.record_id = records_fts.record_id
                    WHERE records_fts MATCH ?1{where_tail}
                    ORDER BY bm25(records_fts), r.collection, r.record_id
                    LIMIT ?{}
                    "#,
                    all_params.len(),
                ),
                all_params,
                title_score_result,
            )
            .await
    }
}

/// Builds SQL filters for collection and metadata predicates.
fn sql_filter(
    alias: &str,
    collections: &[String],
    filter: Option<&MetadataFilter>,
) -> (String, Vec<Value>) {
    let mut clauses = Vec::new();
    let mut params = Vec::new();
    if !collections.is_empty() {
        let placeholders = (0..collections.len())
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(", ");
        clauses.push(format!("{alias}.collection IN ({placeholders})"));
        params.extend(collections.iter().cloned().map(Value::Text));
    }
    if let Some(filter) = filter {
        append_metadata_filter(alias, filter, &mut clauses, &mut params);
    }

    if clauses.is_empty() {
        return (String::new(), params);
    }

    (format!(" WHERE {}", clauses.join(" AND ")), params)
}

/// Appends metadata predicates compiled to SQL.
fn append_metadata_filter(
    alias: &str,
    filter: &MetadataFilter,
    clauses: &mut Vec<String>,
    params: &mut Vec<Value>,
) {
    match filter {
        MetadataFilter::Equals { key, value } => {
            // Metadata lives as exact key/type/value tokens so SQL prefilters
            // rows without needing dynamic JSON paths or arbitrary columns.
            clauses.push(format!("instr({alias}.metadata_tokens, ?) > 0"));
            params.push(Value::Text(metadata_token(key, value)));
        }
        MetadataFilter::All(filters) => {
            for filter in filters {
                append_metadata_filter(alias, filter, clauses, params);
            }
        }
    }
}

/// Serializes metadata for record hydration.
fn metadata_json(metadata: &Metadata) -> Result<String, VectorError> {
    serde_json::to_string(metadata).map_err(|source| {
        VectorError::Serialization {
            source: Box::new(source),
        }
    })
}

/// Deserializes metadata from storage.
fn parse_metadata(value: String) -> Result<Metadata, VectorError> {
    serde_json::from_str(&value).map_err(|source| {
        VectorError::Deserialization {
            source: Box::new(source),
        }
    })
}

/// Encodes metadata as exact-search tokens.
fn metadata_tokens(metadata: &Metadata) -> String {
    metadata
        .iter()
        .map(|(key, value)| metadata_token(key, value))
        .collect::<Vec<_>>()
        .join("")
}

/// Encodes one metadata token.
fn metadata_token(key: &str, value: &MetadataValue) -> String {
    let (kind, value) = match value {
        MetadataValue::Text(value) => {
            ("text", crate::hex::encode(value.as_bytes()))
        }
        MetadataValue::Integer(value) => ("integer", value.to_string()),
        MetadataValue::Float(value) => {
            ("float", normalized_float(*value).to_string())
        }
        MetadataValue::Boolean(value) => ("boolean", value.to_string()),
    };
    format!(
        "\u{1f}{}\u{1e}{kind}\u{1e}{value}\u{1f}",
        crate::hex::encode(key.as_bytes())
    )
}

/// Normalizes float metadata for exact token matching.
fn normalized_float(value: f64) -> f64 {
    if value == 0.0 {
        return 0.0;
    }

    value
}

/// Serializes an embedding for libSQL vector functions.
fn embedding_json(embedding: &[f32]) -> Result<String, VectorError> {
    serde_json::to_string(embedding).map_err(|source| {
        VectorError::Serialization {
            source: Box::new(source),
        }
    })
}

/// Builds an FTS query from user text.
fn fts_query(text: &str) -> String {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|term| !term.is_empty())
        .map(|term| format!("\"{term}\""))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Deserializes an embedding from `vector_extract` output.
fn parse_embedding(value: String) -> Result<Vec<f32>, VectorError> {
    serde_json::from_str(&value).map_err(|source| {
        VectorError::Deserialization {
            source: Box::new(source),
        }
    })
}

/// Builds a chunk scored result.
fn chunk_score_result(row: &Row) -> Result<ChunkSearchResult, VectorError> {
    Ok(ChunkSearchResult {
        chunk_id: text_value(row, 0)?,
        record_id: text_value(row, 1)?,
        collection: text_value(row, 2)?,
        text: text_value(row, 3)?,
        metadata: parse_metadata(text_value(row, 4)?)?,
        score: real_value(row, 5)? as f32,
    })
}

/// Builds a title scored result.
fn title_score_result(row: &Row) -> Result<RecordSearchResult, VectorError> {
    Ok(RecordSearchResult {
        record_id: text_value(row, 0)?,
        collection: text_value(row, 1)?,
        title: text_value(row, 2)?,
        metadata: parse_metadata(text_value(row, 3)?)?,
        score: real_value(row, 4)? as f32,
    })
}

/// Builds a record entry from a storage row.
fn row_to_record(row: &Row) -> Result<RecordEntry, VectorError> {
    Ok(RecordEntry {
        record_id: text_value(row, 0)?,
        collection: text_value(row, 1)?,
        title: text_value(row, 2)?,
        title_embedding: parse_embedding(text_value(row, 3)?)?,
        content: text_value(row, 4)?,
        metadata: parse_metadata(text_value(row, 5)?)?,
    })
}

/// Reads a text column.
fn text_value(row: &Row, index: usize) -> Result<String, VectorError> {
    let value = row.get_value(column_index(index)?).map_err(query_error)?;
    match value {
        Value::Text(value) => Ok(value),
        Value::Blob(value) => String::from_utf8(value).map_err(|source| {
            VectorError::Deserialization {
                source: Box::new(source),
            }
        }),
        _ => Err(VectorError::Deserialization {
            source: invalid_data("expected text column"),
        }),
    }
}

/// Reads a real-valued column.
fn real_value(row: &Row, index: usize) -> Result<f64, VectorError> {
    match row.get_value(column_index(index)?).map_err(query_error)? {
        Value::Real(value) => Ok(value),
        Value::Integer(value) => Ok(value as f64),
        _ => Err(VectorError::Deserialization {
            source: invalid_data("expected numeric column"),
        }),
    }
}

/// Converts a column index into libSQL's index type.
fn column_index(index: usize) -> Result<i32, VectorError> {
    i32::try_from(index).map_err(|source| VectorError::Deserialization {
        source: Box::new(source),
    })
}

/// Validates embedding dimensions.
fn checked_dimensions(dimensions: usize) -> Result<(), VectorError> {
    if dimensions > 0 && i32::try_from(dimensions).is_ok() {
        return Ok(());
    }

    Err(VectorError::InvalidDimensions { dimensions })
}

/// Converts a limit into a SQL integer.
fn limit_i64(limit: usize) -> Result<i64, VectorError> {
    i64::try_from(limit).map_err(|source| VectorError::Serialization {
        source: Box::new(source),
    })
}

/// Builds the metadata table target.
fn meta_target() -> VectorTarget {
    VectorTarget::Table {
        name: "vector_meta",
    }
}

/// Builds a record target.
fn record_target(collection: &str, record_id: &str) -> VectorTarget {
    VectorTarget::Record {
        collection: collection.to_string(),
        record_id: record_id.to_string(),
    }
}

/// Builds a chunks target.
fn chunks_target(collection: &str, record_id: &str) -> VectorTarget {
    VectorTarget::Chunks {
        collection: collection.to_string(),
        record_id: record_id.to_string(),
    }
}

/// Converts a database init error.
fn db_init_error(source: libsql::Error) -> VectorError {
    VectorError::DbInit {
        engine: ENGINE_NAME,
        source: Box::new(source),
    }
}

/// Converts a database write error.
fn write_error(
    source: libsql::Error,
    operation: VectorWriteOperation,
    target: VectorTarget,
) -> VectorError {
    VectorError::WriteFailed {
        engine: ENGINE_NAME,
        operation,
        target,
        source: Box::new(source),
    }
}

/// Converts a database query error.
fn query_error(source: libsql::Error) -> VectorError {
    VectorError::QueryExecution {
        engine: ENGINE_NAME,
        source: Box::new(source),
    }
}

/// Converts a deserialization error.
fn deserialization_error(
    source: impl std::error::Error + Send + Sync + 'static,
) -> VectorError {
    VectorError::Deserialization {
        source: Box::new(source),
    }
}

/// Builds an invalid-data source error.
fn invalid_data(
    message: &'static str,
) -> Box<dyn std::error::Error + Send + Sync> {
    Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        message,
    ))
}
