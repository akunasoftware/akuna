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
  needs). It returns RAW logits and has no normalize parameter (that flag
  belongs to `rerank_with_options`, which re-sorts a document list — not
  the right fit here): the pipeline applies `crate::ml::sigmoid_f32`
  (pub(crate), reachable) to the outputs itself. `score_batch` is sync
  Burn inference — do not call it inline on the async path; wrap it the
  same way `Index` wraps its other sync inference calls (e.g.
  `spawn_blocking`).
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
2. **Fusion (RRF, k = 60, ranks start at 1).** First collapse each
   chunk-level list to record rank — order records by their best chunk and
   re-rank densely 1..n — so long records can't win by chunk count; title
   lists are already record-level. Then fused(record) = Σ over lists
   containing it of 1/(k + rank); absence from a list contributes nothing.
   A "title hit" = the record appeared in a title list's returned
   candidates (no score threshold). Carry evidence forward per record: its
   retrieved chunk texts (deduped by chunk id across lists) and the
   title-hit flag.
3. **First rerank** (when `reranking_model` is set). Evidence set per
   record = retrieved chunk texts PLUS the title when it had a title hit
   (a strong title match must survive for chunk-matched records too). One
   `score_batch` pass over all (query, evidence) pairs, sigmoid applied —
   user-facing scores are 0–1. Roll up: record score = max over its
   evidence scores; best evidence = the argmax. Reranker disabled → the
   RRF fused score is the record score, and best evidence = the chunk
   with the best collapsed record rank (ties: dense list before BM25), or
   the title for title-only hits (document: scores are only comparable
   within one query, and change scale entirely with the reranker toggle).
4. **Expansion seam.** A private stage between roll-up and final ordering,
   passthrough in this step; step 07 replaces only its body. PINNED
   CONTRACT: the stage input/output candidate type carries per record —
   id, collection, title, metadata, score, and KIND-TAGGED best evidence:
   `Chunk(text)` | `Title` | `LeadingWindow(text)` (07 adds the last) |
   none. Steps 07 and 08 branch on the kind (07 must not concatenate a
   title with itself; 08's fallback triggers on `Title`/none) — a bare
   string is not enough, and comparing text against the title is wrong (a
   chunk can legitimately equal it). Do not drop evidence in any stage.
5. **Results.** Validation order: trimmed-empty `text` errors first;
   then `limit == 0` returns empty. Order by score descending
   (`f32::total_cmp`), tie-break ascending on `(collection, record_id)`
   for deterministic parity tests, truncate to `limit`. Hydrate anything
   the candidate doesn't already carry with ONE batch `get_records` call,
   retaining `content` (step 08's fallback needs it even though results
   don't) — result assembly does no graph reads (step 07's traversal is
   the pipeline's only graph consumer). `preview: None`.

FFI: mirror `IndexSearchQuery`/`IndexSearchResult` — and `MetadataFilter`,
whose mirror was deferred from step 05 to land with its first consumer
(smoke-test bindgen on the recursive enum early; escalate if it chokes).
Export `search` as an async method; extend `test_parity_index.py`:
dense-only config (`fulltext: false`), full config, filtered,
multi-collection, title-match retrieval, empty index, empty query text
raises.

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
