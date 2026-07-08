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
- Backend: Grafeo (`storage/graph/backend/`), pulled in with
  `hybrid-search`, `text-index`, `vector-index` features in
  `src-crates/core/Cargo.toml`.
- Consumer: `src-crates/app/src/api/knowledge/mod.rs` uses `search_nodes`
  and embeds node text for `put_node` (`/graph/nodes/search` route).

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

Direction handling (outgoing/incoming/both) is the implementer's call —
pick what expansion needs (both, most likely) and keep the surface minimal.
Multi-hop stays out; step 07 composes hops if it wants depth.

Backend/deps:

- Update the Grafeo backend accordingly; drop the `hybrid-search`,
  `text-index`, and `vector-index` grafeo features if the crate builds
  without them.

App impact (deliberate, decided by owner):

- Remove the `/graph/nodes/search` route and the embedding wiring in
  `src-crates/app/src/api/knowledge/mod.rs`, and update its tests and the
  OpenAPI registrations in `api/server.rs`. Knowledge search returns in
  step 09 built on `Index`. Interim loss of that endpoint is accepted —
  greenfield rules apply.

Node semantics going forward (matters for step 05):

- Records will be stored as nodes with `name` = record title and full
  record content inside the node `metadata` payload — NOT in `description`.
  Nothing to implement here; do not design node search back in.

## Scope

- Trait simplification, traversal method, Grafeo backend update, feature
  slimming, app knowledge API cleanup, test updates.

## Out of scope

- Record-specific graph shapes (step 05 maps records onto the generic
  node/edge types).
- Any replacement search endpoint (step 09).

## Acceptance

- `./build/scripts/ws-check.sh` and `./build/scripts/ws-test.sh` pass.
- No search-related items remain anywhere in `storage/graph`.
- `neighbors` covered by module tests (including nodes with no edges).
