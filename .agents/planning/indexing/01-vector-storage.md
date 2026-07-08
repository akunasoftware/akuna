# Step 01 — Vector Storage

Read `00-overview.md` first. No prior steps required.

## Goal

Create `storage/vector`: an async search-storage trait plus a LanceDB
backend that stores chunk rows and record rows with embeddings, content,
and metadata, and answers dense vector search with collection/metadata
filtering pushed into the engine.

This layer owns ALL searching for `Index` (dense now, BM25 in step 02) and
is the authoritative store for record content. It is core-internal
infrastructure: public in core for `Index` to compose, never exposed over
FFI.

## Context

- `storage/graph` (`src-crates/core/src/storage/graph/`) is the structural
  pattern for module layout: `mod.rs` with domain types + trait, `backend/`
  with the concrete implementation, `error.rs`. (It has no tests — the
  module-test convention to follow is `#[cfg(test)] mod tests;` with a
  sibling `tests.rs`, as in `embedding/` and `reranking/`.)
- Core features live in `src-crates/core/Cargo.toml`; graph storage is the
  `storage` feature. LanceDB and its supporting deps go under the same
  feature. The vector backend needs `tokio` (LanceDB's API is async) and a
  temp-dir crate (`tempfile`) — both under `storage`.
- LanceDB: pin `lancedb` 0.31.x deliberately (verified 2026-07: embedded
  in-process usage, MSRV 1.91 vs our 1.96, arrow ^58 matching the arrow
  58.x already in-tree via grafeo — no dual arrow stack; it adds
  DataFusion, an accepted compile-weight decision). It is pure embedded
  Rust — no external runtime, principles-compliant. Build-time note:
  building lance requires `protoc` on the build machine; ensure CI/dev
  images have it (nix devshell: add if missing).

## Design

New module `src-crates/core/src/storage/vector/`:

- `mod.rs` — domain types + `VectorDbContext` trait + constructors
- `backend/` — LanceDB implementation
- `error.rs` — `VectorError` mirroring `GraphError`'s granularity
  (engine-tagged variants with `source`), plus a dimension-mismatch variant
- `tests.rs` — module tests

Shared metadata types at the `storage` root (`storage/mod.rs`) — both
engines and `Index` reuse these; one shape per concept:

```rust
pub enum MetadataValue { Text(String), Integer(i64), Float(f64), Boolean(bool) }
pub type Metadata = BTreeMap<String, MetadataValue>;
pub enum MetadataFilter {
    Equals { key: String, value: MetadataValue },  // key equals value
    All(Vec<MetadataFilter>),                      // every condition holds
}
```

Rows (two tables). Field lists are REQUIRED — later steps consume them:

```rust
pub struct ChunkEntry {
    pub chunk_id: String,   // unique within its record: "{record_id}:{sequence}"
    pub record_id: String,
    pub collection: String,
    pub sequence: u32,
    pub text: String,
    pub embedding: Vec<f32>,
    pub metadata: Metadata, // identical to the record's metadata
}

pub struct RecordEntry {
    pub record_id: String,
    pub collection: String,
    pub title: String,
    pub title_embedding: Vec<f32>,
    pub content: String,    // full record content — authoritative store
    pub metadata: Metadata,
}
```

Trait — async methods (LanceDB is async; `Index` is async). Use the
async-trait pattern that keeps `Box<dyn VectorDbContext>` object-safe:

```rust
pub trait VectorDbContext: Send + Sync {
    /// Replaces all chunk rows for one record. Empty slice clears them.
    async fn put_chunks(&self, collection: &str, record_id: &str, chunks: &[ChunkEntry]) -> Result<(), VectorError>;
    /// Stores or replaces the record row.
    async fn put_record(&self, record: &RecordEntry) -> Result<(), VectorError>;
    /// Deletes all rows for a record from both tables. Idempotent.
    async fn delete_record(&self, collection: &str, record_id: &str) -> Result<(), VectorError>;
    /// Reads one record row.
    async fn get_record(&self, collection: &str, record_id: &str) -> Result<Option<RecordEntry>, VectorError>;
    /// Reads many record rows in one call (result assembly, expansion, previews).
    async fn get_records(&self, keys: &[(String, String)]) -> Result<Vec<RecordEntry>, VectorError>;
    /// Dense search over chunk embeddings.
    async fn search_chunks(&self, query: &VectorSearchQuery) -> Result<Vec<ChunkSearchResult>, VectorError>;
    /// Dense search over title embeddings.
    async fn search_titles(&self, query: &VectorSearchQuery) -> Result<Vec<RecordSearchResult>, VectorError>;
}
```

Query and results — field lists REQUIRED (step 06 builds record-level
results from these without touching the graph):

```rust
pub struct VectorSearchQuery {
    pub embedding: Vec<f32>,
    pub collections: Vec<String>,      // empty = all
    pub filter: Option<MetadataFilter>,
    pub limit: usize,                  // chunk search: chunk rows; title search: records
}
pub struct ChunkSearchResult { pub chunk_id: String, pub record_id: String, pub collection: String, pub text: String, pub metadata: Metadata, pub score: f32 }
pub struct RecordSearchResult { pub record_id: String, pub collection: String, pub title: String, pub metadata: Metadata, pub score: f32 }
```

Pinned semantics:

- **Scores are similarities**: `f32`, higher = better (cosine).
- **Filtering executes inside the engine** (LanceDB filter pushdown), never
  post-hoc in Rust. Arbitrary metadata keys cannot be static Arrow columns;
  the encoding is the implementer's design (a workable scheme: a
  `List<Utf8>` column of `key/type/value` tokens compiled to
  `array_contains` predicates, `All` → SQL `AND`). Requirements: filtering
  stays in-engine, the scheme is documented as a `//` comment, and the
  table schema tolerates step 02 adding FTS indexes without migration.
- **Embedding dimension is fixed at open** and validated: writes with the
  wrong dimension error; reopening a persistent store whose stored
  dimension differs errors at open.
- ANN index creation may be lazy/best-effort (LanceDB needs a minimum row
  count); flat search below the threshold is fine. Correctness over tuning.

Constructors:

```rust
pub async fn open_context(path: impl AsRef<Path>, dimensions: usize) -> Result<Box<dyn VectorDbContext>, VectorError>;
pub async fn in_memory_context(dimensions: usize) -> Result<Box<dyn VectorDbContext>, VectorError>;
```

(Step 02 adds an index-config options parameter to both — changing these
signatures in place then is intended; greenfield.)

Ephemeral note: LanceDB's Rust SDK does have `connect("memory://")`, but it
is undocumented/unstable surface (lancedb#676, Windows-path bug #1051). We
deliberately back `in_memory_context` with a temp directory owned by the
context and removed on drop — it also keeps the layout identical to
persistent mode.

## Scope

- Trait, types, errors, LanceDB backend, dense search, in-engine filtering,
  dimension handling, batch hydration (`get_records`).
- Update `storage/mod.rs` module docs and re-exports for the new module.
- Module tests: CRUD round-trip (content survives), filtered search
  (`Equals` and `All`), multi-collection vs scoped search,
  replace-all-chunks (old chunks gone; empty slice clears), idempotent
  delete, dimension mismatch errors, in-memory lifecycle (temp dir removed
  on drop).

## Out of scope

- BM25/full-text (step 02).
- Embedding generation — callers pass embeddings in.
- `Index`, FFI, chunking, any graph changes.

## Acceptance

- `./build/scripts/ws-check.sh` and `./build/scripts/ws-test.sh` pass
  (ws-check runs `cargo deny` and `cargo machete` — the new deps must pass
  license checks and be used).
- Docstrings on all public items — purpose, never mechanics.
- No public surface beyond what `Index` will need.
