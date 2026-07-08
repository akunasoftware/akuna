# Step 07 — Graph Expansion

Read `00-overview.md` first. Requires step 06.

## Goal

Fill the expansion seam in the search pipeline: traverse relationships from
candidate records to pull in related records, then produce the final
aggregate ranking. This is what makes results a holistic packet rather than
isolated hits.

## Context

- Search pipeline (06) with a passthrough expansion stage whose candidate
  type carries id, collection, title, metadata, score, and KIND-TAGGED
  best evidence (`Chunk(text)` | `Title` | `LeadingWindow(text)` | none)
  per record — pinned contract; preserve it through this stage.
- `GraphDbContext::neighbors(labels, id)` (04): both directions, one hop,
  `Ok(vec![])` for missing/edge-less nodes.
- Record nodes (05): `labels = ["Record", collection]`, `name` = title,
  payload = metadata + collection, NO content in graph. Content and other
  record-row data hydrate from the vector layer (`get_records`).
- Relationships are caller-supplied (parent/cites/…); extraction adds
  file-path links soon, NER later — assume no particular predicate
  vocabulary.

## Design

Expansion (runs only when `graph: true`; stage is a no-op otherwise):

1. Seeds: the top `limit` candidates by roll-up score (named constant
   logic; documented).
2. For each seed, `neighbors(&["Record", collection], id)` — one hop, no
   depth knob (do not add one speculatively). Expanded records are NOT
   themselves expanded.
3. Map each neighbor back to `(collection, record_id)` from its labels/
   payload (collection = the non-`"Record"` label). Enforce the query
   scope caller-side: drop neighbors outside `query.collections` (when
   non-empty) or failing `query.filter`, evaluated via
   `MetadataFilter::matches` (step 01) — the same semantics the engines
   compile to SQL; this is why the graph stores collection + metadata.
4. Dedup on `(collection, record_id)` against candidates and other
   expansions; a record reached from multiple seeds enters once. Cap
   accepted expansions at `2 × limit`: process seeds in descending score
   order, and WITHIN a seed sort neighbors ascending by
   `(collection, record_id)` before taking — the cap cut must be
   deterministic (`neighbors` return order is not). Hub nodes must not
   explode the rerank batch. (Seed-order processing also makes "max parent
   seed score" for damping simply the first seed that reached the record.)
5. Hydrate expanded records (title/content/metadata) with one batch
   `get_records` from the vector layer.

Final aggregation — decision made (owner's call, Option A), with the
inertness carve-out that makes it consistent:

- **If traversal accepted zero expanded records, skip final aggregation
  entirely** — step 06's ordering passes through unchanged. This satisfies
  "expansion is inert when records have no relationships" (no result
  change, no extra rerank cost) at the price of two scoring regimes;
  record the rationale as a `//` comment.
- Otherwise (reranker enabled): build one bounded representative text per
  post-expansion candidate by evidence kind — `Chunk`: title + chunk
  text; `Title` or none: the title alone (never concatenate the title
  with itself); expanded records (`LeadingWindow`): title + leading
  content window (~1500 chars, from hydration). One `score_batch` pass,
  sigmoid applied (same mechanics as 06 — `score_batch` has no normalize
  flag), scores retrieved and expanded records uniformly; those scores
  become final.
- Reranker disabled: retrieved records keep their step 06 scores; expanded
  records inherit `0.5 × max(parent seed scores)` (named damping
  constant).
- After scoring: sort, tie-break, truncate to `limit` — the step 06
  results contract stands; expanded records compete for the same `limit`
  slots (acceptance tests use `limit ≥ 2` so a seed and its neighbor can
  both appear).
- Expanded records carry `LeadingWindow(text)` as their best evidence
  (step 08's input); they have no chunk evidence.

Expanded records become ordinary `IndexSearchResult`s — no flag
distinguishing them.

## Scope

- Expansion stage, final aggregation as pinned above, dedup, caller-side
  filter enforcement, hydration, core tests: linked records surface
  (`limit ≥ 2`); filters prune expansion; collection scope prunes
  expansion; `graph: false` skips the stage; cycles/self-links terminate;
  no-relationship searches return step 06's exact ordering; hub-node cap
  holds; reranker-off damping.
- FFI surface is unchanged (no new options) — extend the Python parity
  search test with one relationship case; `ws-parity.sh` still must pass.

## Out of scope

- NER, entity nodes, multi-hop traversal, edge weighting, depth options.

## Acceptance

- `./build/scripts/ws-check.sh`, `ws-test.sh`, `ws-parity.sh` pass.
- A search whose best hit links to a related record returns both (with
  `limit ≥ 2`), ranked sensibly.
- A search over records with no relationships returns byte-identical
  results to step 06.
