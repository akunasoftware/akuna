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

- `storage/vector` from step 01: `VectorDbContext` trait, LanceDB backend,
  chunk and title tables.
- LanceDB provides native full-text (BM25) indexes over text columns.

## Design

Extend `VectorDbContext`:

```rust
/// BM25 search over chunk text.
fn search_chunks_text(&self, query: &TextSearchQuery) -> Result<Vec<ChunkSearchResult>, VectorError>;
/// BM25 search over record titles.
fn search_titles_text(&self, query: &TextSearchQuery) -> Result<Vec<TitleSearchResult>, VectorError>;
```

`TextSearchQuery` carries: query text, collections (empty = all), optional
`MetadataFilter`, limit. Result types are the same as dense search — one
shape per concept; only the scoring source differs.

FTS index creation is controlled at context open (the vector layer takes an
options input saying which indexes to build), because `IndexOptions` decides
at init which retrieval functions exist. Opening a persistent context must
handle both fresh creation and reopening with existing indexes.

Scores: BM25 scores and cosine distances are not comparable. Do NOT try to
normalize here — fusion is the search pipeline's job (step 06). Return raw
engine scores.

## Scope

- Both FTS methods, index creation/reopen handling, index-config input on
  the open constructors.
- Tests: lexical matches on chunk text and titles, filtered FTS, behavior
  when FTS was not enabled at open (typed error, not panic).

## Out of scope

- Score fusion, ranking, reranking (step 06).
- Any public API outside `storage/vector`.

## Acceptance

- `./build/scripts/ws-check.sh` and `./build/scripts/ws-test.sh` pass.
- Docstrings on all public items.
- Dense-only contexts from step 01 keep working unchanged.
