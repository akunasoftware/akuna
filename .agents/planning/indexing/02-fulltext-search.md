# Step 02 — Full-Text Search

Read `00-overview.md` first. Requires step 01 (vector storage).

## Goal

Add BM25 full-text search to the vector layer: over chunk text and over
record titles. Same trait, same backend, same filtering semantics as dense
search.

Title search exists because titles are a strong record-level relevance
signal; title matches join the candidate pool before the first rerank
(pipeline wiring happens in step 06 — this step only provides the search
capability).

## Context

- `storage/vector` from step 01: `VectorDbContext`, LanceDB backend, chunk
  and record tables.
- LanceDB's Rust SDK has native BM25 FTS: `Index::FTS` /
  `FtsIndexBuilder` for index creation, `QueryBase::full_text_search` for
  querying. Available since lancedb 0.10, no cargo feature flag needed
  (verified 2026-07).

## Design

Extend `VectorDbContext` (async, like the rest):

```rust
/// BM25 search over chunk text.
async fn search_chunks_text(&self, query: &TextSearchQuery) -> Result<Vec<ChunkSearchResult>, VectorError>;
/// BM25 search over record titles.
async fn search_titles_text(&self, query: &TextSearchQuery) -> Result<Vec<RecordSearchResult>, VectorError>;
```

```rust
pub struct TextSearchQuery {
    pub text: String,
    pub collections: Vec<String>,      // empty = all
    pub filter: Option<MetadataFilter>,
    pub limit: usize,                  // chunk search: chunk rows; title search: records
}
```

Result types are step 01's, unchanged — one shape per concept;
only the scoring source differs. Scores are raw BM25 (do NOT normalize or
try to make them comparable with cosine similarities — fusion is step 06's
job and is rank-based).

Index configuration at open. Both constructors gain an options parameter —
an intended in-place signature change to step 01's constructors
(greenfield, no parallel constructors):

```rust
#[derive(Clone, Debug, Default)]   // Default = dense-only (step 01 behavior)
pub struct VectorContextOptions {
    pub chunk_text_index: bool,    // BM25 over chunk text
    pub title_text_index: bool,    // BM25 over titles
}
pub async fn open_context(path: impl AsRef<Path>, dimensions: usize, options: &VectorContextOptions) -> ...;
pub async fn in_memory_context(dimensions: usize, options: &VectorContextOptions) -> ...;
```

Pinned behaviors:

- **Open handles fresh and reopen.** Fresh store with a flag on: create the
  FTS index on the relevant text column. Reopen: create only what is
  missing; existing indexes reused. Opening an empty store must not fail —
  if the pinned lancedb version cannot create an FTS index before data
  exists, create lazily on first write; the API behavior must be
  indistinguishable either way.
- **Writes stay searchable.** LanceDB FTS indexes do not automatically
  cover rows written after index creation; the backend must keep searches
  seeing current data (table `optimize()` after writes, or the version's
  unindexed-tail scan). A test pins it: rows written after open are found.
- **Gating is context state, not engine probing.** The context records
  which indexes were enabled; calling an FTS search on a context opened
  without that index returns a typed `VectorError` variant (one variant +
  target enum, mirroring `GraphError`'s `GraphTarget` pattern) — never a
  panic, never silently empty.
- Flag mismatches on reopen are lenient here (create missing, ignore
  extra); strict config compatibility is the step 05 manifest's job.
- Tokenizer/language config: LanceDB defaults, nothing exposed
  ("configure only what genuinely warrants it").

Verify before building: that `full_text_search` composes with the metadata
predicate as a PRE-filter in the pinned lancedb version (FTS + filter had
post-filter-only limitations historically — if pre-filtering is impossible
on this path, escalate to the owner rather than silently post-filtering);
and which branch applies for index creation on an empty table.

## Scope

- Both FTS methods, `VectorContextOptions`, constructor change, index
  create/reopen handling, write-freshness handling.
- Tests: lexical matches on chunk text and titles; filtered FTS (collection
  + metadata); multi-collection; typed error when FTS disabled; persistent
  reopen keeps FTS working; rows written after open are findable;
  dense-only (default options) contexts still pass all step 01 tests.

## Out of scope

- Score fusion, ranking, reranking (step 06).
- Any public API outside `storage/vector`.

## Acceptance

- `./build/scripts/ws-check.sh` and `./build/scripts/ws-test.sh` pass.
- Docstrings on all public items.
- Dense-only behavior is byte-for-byte what step 01 shipped, modulo the
  constructor signatures.
