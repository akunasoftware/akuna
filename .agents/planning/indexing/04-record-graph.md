# Step 04 — Graph Simplification & Traversal

Read `00-overview.md` first. No prior steps required (independent of 01–03).

## Goal

Make the graph layer a pure graph: nodes, edges, traversal. Remove all
string/vector search from its API — search is too backend-unique a feature
to demand from graph engines, and the vector layer (steps 01–02) owns
searching. Add the traversal query that graph expansion (step 07) needs.

## Context

- `src-crates/core/src/storage/graph/mod.rs` — `GraphDbContext` trait with
  `put_node(node, search_embedding)`, `search_nodes`, plus
  `GraphNodeSearchQuery`, `GraphNodeSearchResult`, `search_text`.
- `src-crates/core/src/storage/mod.rs` re-exports the search types and has
  a doctest calling `put_node(&node, &[])` — both break with this change;
  update them (module docs too).
- Backend: Grafeo (`storage/graph/backend/grafeo.rs`). There are currently
  NO tests anywhere under `storage/graph` — traversal tests are net-new.
- Consumer: `src-crates/app/src/api/knowledge/mod.rs` uses `search_nodes`
  and embeds node text for `put_node` (`/graph/nodes/search` route), with
  OpenAPI registrations in `src-crates/app/src/api/server.rs`.

## Design

Trait changes:

- `put_node(&self, node: &GraphNode)` — drop the `search_embedding`
  parameter.
- Delete `search_nodes`, `GraphNodeSearchQuery`, `GraphNodeSearchResult`,
  `search_text`.
- Add traversal:

```rust
/// Nodes connected to the given node, with the edges that connect them.
fn neighbors(&self, labels: &[&str], id: &str) -> Result<Vec<(GraphEdge, GraphNode)>, GraphError>;
```

Pinned traversal semantics:

- Direction: both — union of outgoing and incoming edges; each returned
  `GraphEdge` keeps its stored direction (callers compare the queried id
  against `edge.source`/`edge.target`). No direction parameter.
- Missing start node → `Ok(vec![])`, same as a node with no edges (pure
  query, mirrors `get_node`'s tolerance — deliberately unlike
  `delete_node`'s `NotFound`).
- One tuple per edge; no dedup of repeated neighbor nodes.
- Single hop only; step 07 composes hops if it ever wants depth.

Backend notes:

- Remove from the Grafeo backend: the `search_nodes` impl,
  `ensure_search_indexes`, the search-text/embedding property writes, and
  the text/vector index creation calls. Keep `_id` reserved-prefix
  handling.
- Implementation risk to expect: `neighbors` is the first operation that
  reads edges and node labels back out of grafeo. If GQL result rows don't
  carry the relationship predicate or node labels, hydrate via grafeo's
  node API (as the old search path did) — either strategy is fine; pick
  what works and keep it internal.

Dependency slimming (corrected — the naive version is a no-op):

- Grafeo's `embedded` feature transitively enables `ai` =
  `vector-index` + `text-index` + `hybrid-search` + `cdc`, so merely
  deleting those three names from our feature list changes nothing.
- Replace `embedded` with a curated minimal set that keeps LPG + GQL +
  persistence working (candidates: `edge`/`lpg`, `gql`, `grafeo-file`,
  `algos`, `parallel`, `regex`, `arrow-export` — verify by build+test). If
  no curated set builds cleanly, keep `embedded` — the search features
  pull no external crates, so the win is compile-time only; don't fight
  for it.
- Drop `rdf` regardless (sparql/shacl/graphql machinery, nothing uses it)
  and the redundant explicit `hybrid-search`/`text-index`/`vector-index`
  entries.

App impact (deliberate, decided by owner):

- Remove the `/graph/nodes/search` route, `search_nodes` handler,
  `NodeSearchQuery`, and both `embed_search_text` variants (prod OnceCell
  + test stub) from the knowledge module; drop the `knowledge::search_nodes`
  path and `GraphNodeSearchResult` schema from `server.rs`. `write_node`
  loses its embedding call (and its reason to be async). Remaining
  node/edge CRUD routes and tests stay. Knowledge search returns in step
  09 built on `Index`; the interim gap is accepted — greenfield rules.

Node semantics going forward (context for steps 05/07 — nothing to build
here): records will be stored as nodes with `labels = ["Record", collection]`,
`name` = title, and record metadata + collection in the node payload.
Content does NOT live in the graph — the vector layer's record row is the
authoritative content store. Do not design node search or content storage
back in.

## Scope

- Trait simplification, traversal method + pinned semantics, Grafeo
  backend update, `storage/mod.rs` re-export/doctest/doc fixes, feature
  slimming as corrected above, app knowledge cleanup, net-new graph module
  tests through `in_memory_context()` (neighbors with edges in both
  directions, node with no edges, missing node).

## Out of scope

- Record-specific graph shapes (step 05 maps records onto the generic
  node/edge types).
- Any replacement search endpoint (step 09).

## Acceptance

- `./build/scripts/ws-check.sh` and `./build/scripts/ws-test.sh` pass.
- No search-related items remain anywhere in `storage/graph` (the generic
  `GraphError` variants stay).
- `neighbors` covered by module tests including direction and missing-node
  behavior.
