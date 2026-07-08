# Step 01 — Vector Storage

Read `00-overview.md` first. No prior steps required.

## Goal

Create `storage/vector`: a search-storage trait plus a LanceDB backend that
stores chunks and record titles with embeddings and metadata, and answers
dense vector search with collection/metadata filtering.

This layer owns ALL searching for `Index` (dense now, BM25 in step 02). It is
core-internal infrastructure: public in core for `Index` to compose, no FFI
exposure.

## Context

- `storage/graph` (`src-crates/core/src/storage/graph/`) is the structural
  pattern to mirror: `mod.rs` with domain types + trait, `backend/` with the
  concrete implementation, `error.rs`.
- Core features live in `src-crates/core/Cargo.toml`; graph storage is the
  `storage` feature. Add LanceDB under the same feature.
- The LanceDB Rust crate is embedded (no external runtime — principles
  compliant). Its Arrow/DataFusion dependency weight is accepted; this was a
  deliberate decision.

## Design

New module `src-crates/core/src/storage/vector/`:

- `mod.rs` — domain types + `VectorDbContext` trait
- `backend/` — LanceDB implementation
- `error.rs` — `VectorError` mirroring the granularity of `GraphError`
- `tests.rs` — module tests per repo convention

Shared metadata types at the `storage` root (both engines and later `Index`
reuse these — one shape per concept):

```rust
/// Scalar metadata value.
pub enum MetadataValue { Text(String), Integer(i64), Float(f64), Boolean(bool) }

/// Record metadata map.
pub type Metadata = BTreeMap<String, MetadataValue>;

/// Metadata filter. Equality-only today; enum so operators can be added
/// without breaking the surface.
pub enum MetadataFilter {
    /// Key equals value.
    Equals { key: String, value: MetadataValue },
    /// All conditions hold.
    All(Vec<MetadataFilter>),
}
```

Storage rows (exact field layout is the implementer's call; these fields are
required):

- **Chunk row**: chunk id, record id, collection, sequence number, text,
  embedding, metadata.
- **Title row**: record id, collection, title text, title embedding,
  metadata. One row per record.

Trait sketch (adjust signatures as implementation demands, keep the shape):

```rust
pub trait VectorDbContext: Send + Sync {
    /// Replaces all chunk rows for a record.
    fn put_chunks(&self, chunks: &[ChunkEntry]) -> Result<(), VectorError>;
    /// Stores or replaces the title row for a record.
    fn put_title(&self, title: &TitleEntry) -> Result<(), VectorError>;
    /// Deletes all rows for a record.
    fn delete_record(&self, collection: &str, record_id: &str) -> Result<(), VectorError>;
    /// Dense search over chunk embeddings.
    fn search_chunks(&self, query: &VectorSearchQuery) -> Result<Vec<ChunkSearchResult>, VectorError>;
    /// Dense search over title embeddings.
    fn search_titles(&self, query: &VectorSearchQuery) -> Result<Vec<TitleSearchResult>, VectorError>;
}
```

`VectorSearchQuery` carries: query embedding, collections (empty = all),
optional `MetadataFilter`, limit. Filtering happens inside the engine at
query time, not post-hoc in Rust.

Constructors mirror graph storage:

- `open_context(path)` — persistent, rooted at `path`.
- `in_memory_context()` — ephemeral. LanceDB has no pure in-memory mode;
  back this with a temp directory owned by the context and removed on drop.

## Scope

- Trait, types, errors, LanceDB backend, dense search, metadata/collection
  filtering, embedding dimension handling (dimension fixed at context open).
- Module-level tests covering CRUD, filtered search, multi-collection
  search, replace-all-chunks semantics, in-memory lifecycle.

## Out of scope

- BM25/full-text (step 02). Design table schema so FTS indexes can be added
  without migration.
- Embedding generation — callers pass embeddings in.
- `Index` itself, FFI, chunking.

## Acceptance

- `./build/scripts/ws-check.sh` and `./build/scripts/ws-test.sh` pass.
- Docstrings on all public items, purpose-not-mechanics per conventions.
- No public surface beyond what `Index` will need.
