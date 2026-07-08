# Step 05 — Index Core

Read `00-overview.md` first. Requires steps 01–04.

## Goal

Create the `Index` actor: options, opening, collections, and record
add/update/remove/get — wiring together vector storage, graph storage,
chunking, and the embedder. Mirror the new surface in FFI with Python
parity tests. Search lands in step 06.

## Context

- `storage/vector` (01–02): async `VectorDbContext`, LanceDB backend,
  chunk + record rows (record row holds authoritative content), dense +
  FTS, `VectorContextOptions`.
- `storage/graph` (04): simplified sync `GraphDbContext` with `neighbors`.
- `chunking` (03): segmentation + packer, `ChunkingOptions`.
- `embedding::TextEmbedder` / `reranking::TextReranker` — existing actors
  (`async fn new(options)` in core).
- FFI conventions: `src-crates/ffi/src/embedding.rs` et al. Note the
  actual pattern: async construction is a FREE async factory function
  (`load_text_embedder`) with `Option<Options>` defaulting — UniFFI cannot
  generate async primary constructors for Python. Parity tests live in
  `src-crates/ffi/tests/python/`; run `./build/scripts/ws-parity.sh`.

## Design

Module `src-crates/core/src/index/` gains `Index`; new cargo feature
`index = ["storage", "embedding", "reranking", "chunking"]`, added to
`full` (the ffi crate consumes core via `full` — without this the mirror
silently can't compile).

```rust
/// Options for [`Index`].
pub struct IndexOptions {
    pub name: String,                           // storage subpath under the data root; default "default"
    pub path: Option<PathBuf>,                  // data root; None = ephemeral (temp root, same layout)
    pub embedding_model: EmbeddingModel,        // used for chunks and titles
    pub reranking_model: Option<RerankingModel>,// None disables; Default = Some(default model)
    pub fulltext: bool,                         // default true — gates both BM25 functions
    pub graph: bool,                            // default true — gates graph storage + expansion
    pub chunking: ChunkingOptions,
    pub cache_dir: Option<PathBuf>,             // HF cache override
}
```

Manual `Default` (reranking ON, `name: "default"`). One data root hosts
many indexes: the storage root for an index is `<path>/<name>` (and
`<temp>/<name>` in ephemeral mode — same layout rule). `name` is
validated at `new`: non-empty, no path separators or traversal — typed
error otherwise. Dense retrieval is always on and has no toggle. Disabled
functions create no storage/indexes: `graph: false` opens no graph
context; `fulltext: false` opens the vector context without FTS indexes.

Record shapes (public; FFI mirrors 1:1). These are HTTP/OpenAPI surface in
step 09, so derive `Serialize`/`Deserialize` + `utoipa::ToSchema` here
(precedent: `GraphNode`). Pin the wire format of `MetadataValue` AND
`MetadataFilter` to externally-tagged lowercase variants —
`{"text": "..."}`, `{"integer": 3}`;
`{"equals": {"key": "k", "value": {"text": "v"}}}`,
`{"all": [...]}` — so the JSON contract step 09 serves is stable:

```rust
pub struct Record {
    pub id: String,             // unique within its collection
    pub collection: String,
    pub title: String,
    pub content: String,
    pub metadata: Metadata,
    pub relationships: Vec<RecordRelationship>,
}
pub struct RecordRelationship {
    pub predicate: String,      // e.g. "parent", "cites"
    pub record_id: String,      // target id
    pub collection: String,     // target collection
}
```

`Index` — `Send + Sync`, methods take `&self` (internal write
serialization; app state will hold `Arc<Index>`), all async:

```rust
impl Index {
    pub async fn new(options: IndexOptions) -> Result<Self>;
    pub async fn add(&self, records: Vec<Record>) -> Result<()>;
    pub async fn remove(&self, collection: &str, record_id: &str) -> Result<()>;
    pub async fn get(&self, collection: &str, record_id: &str) -> Result<Option<Record>>;
}
```

Pinned behaviors:

- **Add is upsert.** Updating replaces ALL chunk rows (packer output can
  differ entirely) AND the record's outgoing edges (enumerate via
  `neighbors`, delete edges whose source is this record, rewrite from
  `relationships`); incoming edges from other records are untouched.
  Collections are created implicitly. Batch validation runs BEFORE any
  write: non-empty `relationships` with `graph: false` errors up front, as
  does a collection named `"Record"` (reserved — it is the node label, and
  relationship target collections are rebuilt from labels on read).
- **Write path — two phases, pinned.** Phase 1, per record sequentially:
  pack content (03) → embed chunks + title (one `embed_batch` per record,
  document-mode, no query prompt) → vector `put_chunks` + `put_record` →
  graph `put_node`. On a graph-node failure: best-effort vector rollback
  (`delete_record`), return the error naming the record; earlier records
  stay written. Phase 2, after ALL nodes: write edges (intra-batch
  references work in any order). An edge failure errors naming the record;
  nothing rolls back in phase 2 — retrying the same `add` fully heals
  (every write is replace-style). An edge targeting a record that exists
  nowhere is a typed error naming the offender — rely on the backend's
  put_edge failing and wrap it with record context (verify grafeo actually
  errors on missing nodes before leaning on it); do not invent stub nodes.
- **`remove` semantics:** delete the record's vector rows, its graph node,
  and ALL edges touching it — outgoing AND incoming (verify whether grafeo
  `delete_node` cascades edges; if not, enumerate via `neighbors` and
  `delete_edge` first). Vector first, graph second. Idempotent: removing a
  missing record is `Ok` (unlike the graph trait's `NotFound` — `Index`
  absorbs that).
- **Errors:** `anyhow::Result` per the embedding/reranking convention — no
  error enum. The manifest-mismatch and edge-target messages must be
  distinct and stable enough to assert on in tests.
- **Graph mapping:** node `labels = ["Record", collection]` (record ids are
  only unique per collection — the label carries the collection into node
  identity), `name` = title, payload = record metadata + collection. NO
  content in the graph.
- **`get`:** record row from the vector layer (authoritative for
  title/content/metadata in every config) + relationships from the graph
  when enabled (outgoing edges via `neighbors`, filtered to
  `edge.source == this`; target collection = the target's non-`"Record"`
  label). With `graph: false`, `relationships` comes back empty — the
  matching `add` restriction is validated up front (see write path).
- **Model loading:** `new` loads the embedder and, when configured, the
  reranker eagerly — fail-fast at construction; the reranker sits unused
  until step 06. Ephemeral drop order: storage contexts drop before the
  temp root (struct field order matters).
- **Metadata round-trip:** `Metadata` is a plain map; absent-vs-empty is
  not distinguished — `get` returns an empty map where nothing was stored.
- **Manifest** (every storage root `<path>/<name>`, ephemeral included):
  `manifest.json` with `schema_version` (start 1), `embedding_model`,
  `chunking`, `fulltext`, `graph`. (`name` itself is not in the manifest —
  it IS the subpath.) `reranking_model` is deliberately excluded — it affects no
  stored data and may change freely across reopen. `Index::new` on an
  existing root errors clearly, naming the field, on any mismatch
  (embeddings from different models are not comparable; mixed chunk
  geometries are silently wrong). Missing/corrupt manifest in a non-empty
  root → error. Serialize via the manifest module mapping enums to
  strings — don't force serde derives onto `EmbeddingModel` just for this.
- **Ephemeral mode:** temp data root owned by the `Index`, removed on
  drop, identical layout (`<temp>/<name>/` containing `manifest.json`,
  `vector/`, `graph/`). Use the persistent grafeo path under the temp
  root, not grafeo's in-memory mode — same-layout rule.

FFI (`src-crates/ffi/src/index.rs`, registered in `ffi/src/lib.rs`):

- Mirror `IndexOptions`, `ChunkingOptions`, `Record`, `RecordRelationship`,
  `MetadataValue` 1:1 — annotations and conversions only, including
  `path`/`cache_dir` as `Option<String>` (the existing embedding FFI
  omitting `cache_dir` is a known gap; don't copy it). `Metadata` crosses
  as `HashMap<String, MetadataValue>` (UniFFI has no BTreeMap; the
  conversion lives in ffi `From` impls and is fine — type conversion, not
  behavior). `MetadataFilter`'s mirror waits for step 06, where its first
  FFI consumer (`search`) appears — no dead surface; smoke-test bindgen on
  the recursive enum then, and escalate as a core-surface question if it
  chokes rather than reshaping FFI-side.
- Construction: free async factory `load_index(options: Option<IndexOptions>)`
  under `#[uniffi::export(async_runtime = "tokio")]`, exactly like
  `load_text_embedder`. Methods (`add`, `remove`, `get`) export as async
  methods on the `uniffi::Object` — that `impl` block's
  `#[uniffi::export]` must ALSO carry `async_runtime = "tokio"` (no
  in-repo precedent for async object methods; the existing modules are
  sync-method only).
- Python parity `src-crates/ffi/tests/python/test_parity_index.py`
  (pytest-asyncio is already configured): add/get/remove round-trip on
  default (ephemeral) options; update-replaces semantics; persistence
  across reopen via `tmp_path`; embedding-model mismatch raises;
  relationships round-trip.

## Scope

- Everything above, core module tests on ephemeral mode (CRUD, upsert
  replaces chunks and outgoing edges, batch failure semantics, manifest
  mismatch matrix, `graph: false` and `fulltext: false` configs,
  relationships error with `graph: false`), FFI mirror + parity tests.

## Out of scope

- `search` (step 06) — no placeholder method.
- Preview, expansion, app changes, NER, scanning, extraction-fed input.

## Acceptance

- `./build/scripts/ws-check.sh`, `ws-test.sh`, and `ws-parity.sh` pass.
- Default `IndexOptions` works with zero configuration from Rust and
  Python.
- Chunks appear nowhere in the public or FFI surface.
