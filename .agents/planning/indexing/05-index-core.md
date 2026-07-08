# Step 05 — Index Core

Read `00-overview.md` first. Requires steps 01–04.

## Goal

Create the `Index` actor: options, opening, collections, and record
add/update/remove/get — wiring together vector storage, graph storage,
chunking, and the embedder. Mirror the new surface in FFI with Python
parity tests. Search lands in step 06.

## Context

- `storage/vector` (01–02): `VectorDbContext`, LanceDB backend, dense + FTS.
- `storage/graph` (04): simplified `GraphDbContext` with traversal.
- `index::ChunkingOptions` + internal chunker (03).
- `embedding::TextEmbedder` / `reranking::TextReranker` — existing actors;
  follow their construction style (`async fn new(options)`).
- FFI conventions: see `src-crates/ffi/src/embedding.rs` et al; parity
  tests in `src-crates/ffi/tests/python/`; run via
  `./build/scripts/ws-parity.sh`.

## Design

Module `src-crates/core/src/index/` grows `Index`; core feature gating
follows the existing pattern (`index` feature depending on `storage`,
`embedding`, `reranking`).

```rust
/// Options for [`Index`].
pub struct IndexOptions {
    /// Storage root. None = ephemeral storage.
    pub path: Option<PathBuf>,
    /// Embedding checkpoint used for chunks and titles.
    pub embedding_model: EmbeddingModel,
    /// Reranker checkpoint. None disables reranking. Default: Some(default model).
    pub reranking_model: Option<RerankingModel>,
    /// Enable BM25 full-text retrieval (content and titles). Default: true.
    pub fulltext: bool,
    /// Enable graph storage and expansion. Default: true.
    pub graph: bool,
    /// Content chunking behavior.
    pub chunking: ChunkingOptions,
    /// Optional Hugging Face cache directory override.
    pub cache_dir: Option<PathBuf>,
}
```

Field set may flex during implementation, but the constraints are binding:
flat struct, strong types, `Default` produces a fully working index with
reranking ON, and disabled functions must not create their storage/indexes.

Record shapes (public, reused by FFI 1:1):

```rust
/// A unit of indexed content.
pub struct Record {
    /// Stable identifier, unique within its collection.
    pub id: String,
    /// Collection this record belongs to.
    pub collection: String,
    /// Human-readable title.
    pub title: String,
    /// Full text content.
    pub content: String,
    /// Serializable metadata, filterable at search time.
    pub metadata: Metadata,
    /// Typed links to other records.
    pub relationships: Vec<RecordRelationship>,
}

/// Directed, typed link from the owning record to another record.
pub struct RecordRelationship {
    /// Relationship type, e.g. "parent", "cites".
    pub predicate: String,
    /// Target record id.
    pub record_id: String,
    /// Target record collection.
    pub collection: String,
}
```

`Index` API (async where models are involved):

- `Index::new(options) -> Result<Index>` — loads embedder (and reranker if
  configured), opens/creates storage.
- `add(records: Vec<Record>)` — upsert. Updating replaces ALL chunks for
  the record (no diffing). Collections are created implicitly.
- `remove(collection, record_id)` — removes from every engine.
- `get(collection, record_id) -> Option<Record>` — full record, content
  included, read from the graph node payload (or vector layer when
  `graph: false` — implementer picks the authoritative store per config
  and documents it).

Write path per record: chunk content (per `ChunkingOptions`) → embed chunks
+ title → vector layer rows → graph node (label `Record`, `name` = title,
content + metadata in node payload) → relationship edges. Single-writer
actor; keep writes per record atomic-ish (vector first, graph second;
define and test failure behavior).

Persistence manifest (persistent mode): a manifest file at the storage root
recording schema version, embedding model, chunking options, and enabled
functions. `Index::new` over an existing root errors clearly on
incompatible config (embeddings from different models are not comparable).
Ephemeral mode skips nothing: same layout in a temp root.

FFI (`src-crates/ffi/src/index.rs`):

- Mirror `Index`, `IndexOptions`, `Record`, `RecordRelationship`,
  `Metadata`/`MetadataValue`, `MetadataFilter` 1:1 — annotations and
  conversions only.
- Python parity tests `src-crates/ffi/tests/python/test_parity_index.py`:
  add/get/remove round-trip, update-replaces semantics, persistence across
  reopen, config-mismatch error.

## Scope

- Everything above, module tests in core (use in-memory/ephemeral mode),
  FFI mirror + parity tests.

## Out of scope

- `search` (step 06) — do not add a placeholder method.
- Preview, expansion, app changes, NER, scanning, extraction input.

## Acceptance

- `./build/scripts/ws-check.sh`, `ws-test.sh`, and `ws-parity.sh` pass.
- Default `IndexOptions` works with zero configuration in Rust and Python.
- Chunks appear nowhere in the public or FFI surface.
