# Step 06 — Search Pipeline

Read `00-overview.md` first. Requires step 05.

## Goal

Add `Index::search`: candidate retrieval across enabled functions, fusion,
first rerank, and record-level results. Graph expansion enters in step 07
(leave a clean seam with a pinned contract); preview in step 08 (`preview`
stays `None`). Mirror in FFI with parity tests.

## Context

- `Index` (05) with the async vector layer's dense + FTS search (01–02)
  and hydration (`get_records`), plus the loaded reranker.
- Reranker API: the pair-scoring call is `TextReranker::score_batch`
  (scores (query, text) pairs in order — exactly what evidence scoring
  needs). `rerank_with_options` re-sorts a document list; not the right
  fit here.
- Engines return raw, mutually incomparable scores (cosine similarity vs
  BM25) — fusion is rank-based for exactly that reason.

## Design

Public shapes (serde + `utoipa::ToSchema` like step 05's types — step 09
serves them over HTTP):

```rust
pub struct IndexSearchQuery {
    pub text: String,                   // trimmed-empty => typed error
    pub collections: Vec<String>,       // empty = all
    pub filter: Option<MetadataFilter>, // applied in every retrieval call
    pub limit: usize,                   // Default impl: 10; 0 => Ok(empty)
}
pub struct IndexSearchResult {
    pub record_id: String,
    pub collection: String,
    pub title: String,
    pub metadata: Metadata,
    pub score: f32,
    pub preview: Option<String>,        // None until step 08
}
```

`pub async fn search(&self, query: IndexSearchQuery) -> Result<Vec<IndexSearchResult>>`.

Pipeline:

1. **Candidates.** Embed the query text once. Fan out per enabled
   function — dense chunk + dense title always; BM25 chunk + BM25 title
   when `fulltext` — each with collections + filter pushed down and a
   candidate budget of `4 × limit` floored at 20 (named constants).
2. **Fusion (RRF, k = 60).** First collapse each chunk-level list to
   record rank — a record's rank in that list is its best chunk's rank —
   so long records can't win by chunk count; title lists are already
   record-level. Then fused(record) = Σ over lists of 1/(k + rank).
   Carry evidence forward per record: its retrieved chunk texts (deduped
   by chunk id across lists) and its title when any title list hit.
3. **First rerank** (when `reranking_model` is set). Evidence set per
   record = retrieved chunk texts PLUS the title when it had a title hit
   (a strong title match must survive for chunk-matched records too). One
   `score_batch` call over all (query, evidence) pairs, `normalize: true`
   — user-facing scores are sigmoid 0–1. Roll up: record score = max over
   its evidence scores. Reranker disabled → the RRF fused score is the
   record score (document: scores are only comparable within one query,
   and change scale entirely with the reranker toggle).
4. **Expansion seam.** A private stage between roll-up and final ordering,
   passthrough in this step; step 07 replaces only its body. PINNED
   CONTRACT: the stage input/output candidate type carries per record —
   id, collection, title, metadata, score, and best-evidence text (its
   top-scoring chunk text, or title for title-only hits). Steps 07 and 08
   consume that evidence; do not drop it.
5. **Results.** Order by score descending (`f32::total_cmp`), tie-break on
   `(collection, record_id)` for deterministic parity tests, truncate to
   `limit`. Hydrate anything the candidate doesn't already carry with ONE
   batch `get_records` call — no graph reads in search, ever.
   `preview: None`.

FFI: mirror `IndexSearchQuery`/`IndexSearchResult`, export `search` as an
async method; extend `test_parity_index.py`: dense-only config
(`fulltext: false`), full config, filtered, multi-collection, title-match
retrieval, empty index, empty query text raises.

## Scope

- `search` end to end, core module tests (title-only match surfaces the
  record; title evidence lifts a chunk-matched record; disabled functions
  contribute nothing; filters respected across engines; rerank-off
  ordering; limit 0; determinism), FFI mirror + parity.

## Out of scope

- Expansion behavior (07), preview building (08), pagination, query
  operators, highlighting.

## Acceptance

- `./build/scripts/ws-check.sh`, `ws-test.sh`, and `ws-parity.sh` pass.
- A default index answers a search from Rust and Python with record-level
  results only.
- Rerank-disabled and fulltext-disabled configurations search correctly.
