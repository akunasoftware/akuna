# Step 06 — Search Pipeline

Read `00-overview.md` first. Requires step 05.

## Goal

Add `Index::search`: candidate retrieval across all enabled functions,
fusion, first rerank, and record-level results. Graph expansion enters in
step 07 (leave a clean seam); preview in step 08 (`preview` stays `None`).
Mirror in FFI with parity tests.

## Context

- `Index` (05) with vector layer dense + FTS search (01–02) and reranker.
- `TextReranker::rerank_with_options` scores (query, text) pairs.
- Engines return raw, mutually incomparable scores (cosine vs BM25) —
  fusion must not assume comparable scales.

## Design

Public shapes:

```rust
/// Search request.
pub struct IndexSearchQuery {
    /// Query text.
    pub text: String,
    /// Collections to search. Empty = all collections.
    pub collections: Vec<String>,
    /// Optional metadata filter, applied in every engine.
    pub filter: Option<MetadataFilter>,
    /// Maximum results. Default: a sensible small number (e.g. 10).
    pub limit: usize,
}

/// Ranked record-level search hit.
pub struct IndexSearchResult {
    /// Record id.
    pub record_id: String,
    /// Record collection.
    pub collection: String,
    /// Record title.
    pub title: String,
    /// Record metadata.
    pub metadata: Metadata,
    /// Relevance score.
    pub score: f32,
    /// Semantically relevant excerpt. None until step 08.
    pub preview: Option<String>,
}
```

`search(query: IndexSearchQuery) -> Result<Vec<IndexSearchResult>>`.

Pipeline:

1. **Candidates.** Embed the query once. Run, per enabled function:
   dense chunk search, BM25 chunk search, dense title search, BM25 title
   search. Each retrieves a generous candidate multiple of `limit`
   (implementer picks, e.g. 4×) with collection/metadata filters pushed
   down.
2. **Fusion.** Fuse the ranked lists with Reciprocal Rank Fusion (rank-based,
   sidesteps score-scale mismatch; standard constant k=60). Title hits are
   record-level — fuse them alongside chunk hits by record.
3. **First rerank.** When reranking is enabled, rerank candidate texts
   against the query: chunk candidates by chunk text, title-only candidates
   by title. Roll up to records: a record's score is its best-scoring
   evidence (max). Without a reranker, fused rank order stands.
4. **Expansion seam.** A private pipeline stage between roll-up and final
   ordering that currently passes candidates through unchanged. Step 07
   fills it. Structure it so adding expansion touches only that stage.
5. **Results.** Order by score, truncate to `limit`, build
   `IndexSearchResult` from record info already in storage rows (no graph
   read needed on the hot path). `preview: None`.

Design decisions made here (document in code where non-obvious):

- RRF for fusion; roll-up = max evidence score.
- Chunks stay hidden: nothing chunk-shaped in the result or FFI.
- Post-expansion final scoring is step 07's decision, not this step's.

FFI: mirror `IndexSearchQuery`/`IndexSearchResult`, add
`test_parity_index.py` search cases (dense-only config, full config,
filtered search, multi-collection, title-match retrieval, empty index).

## Scope

- `search` end to end as above, core module tests (including: title-only
  match surfaces the record; disabled functions contribute nothing; filters
  respected across engines), FFI mirror + parity.

## Out of scope

- Graph expansion behavior (07), preview building (08), pagination,
  highlighting, query operators.

## Acceptance

- `./build/scripts/ws-check.sh`, `ws-test.sh`, and `ws-parity.sh` pass.
- A default index answers a search in Rust and Python with record-level
  results only.
- Rerank-disabled and fulltext-disabled configurations search correctly.
