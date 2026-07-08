# Step 09 — App Adoption

Read `00-overview.md` first. Requires step 06; do after 07 and 08 so the
product ships the full pipeline.

## Goal

Rewrite the app's knowledge API on top of `Index`, deleting its bespoke
storage wiring. The plan is done when the product uses `Index` — the
knowledge service is implemented in `app` on top of core, never in core.

## Context

- `src-crates/app/src/api/knowledge/mod.rs`: routes over raw graph
  nodes/edges (`ApiState` holds a `GraphDbContext`); its search route was
  removed in step 04.
- `src-crates/app/src/api/knowledge/tests.rs`: route tests against an
  in-memory backend — the pattern to preserve (ephemeral `Index`).
- OpenAPI registrations live in `src-crates/app/src/api/server.rs`
  (utoipa).
- App config/constants: never hardcode the app name; follow existing
  config patterns for the storage path.

## Design

`ApiState` holds an `Index` (ephemeral in tests, persistent path from app
config in production).

Routes become record-shaped — mirror `Index`'s operations, no more raw
node/edge plumbing:

- `POST /records` — add/update records (body: records per core shapes).
- `GET /records/{collection}/{id}` — full record via `get`.
- `DELETE /records/{collection}/{id}` — remove.
- `GET /records/search` — `Index::search`; query params for text,
  collections, limit; metadata filter in whatever encoding fits the
  existing API error/param conventions.

Route shapes above are directional — follow the app's existing REST
conventions where they conflict. Response bodies reuse core types
(`Record`, `IndexSearchResult`) with utoipa schema derives; one shape per
concept — no API-local copies of core types.

Delete:

- Direct graph endpoints (`/graph/nodes`, `/graph/edges`) and their
  handlers/types — record relationships now travel inside record bodies.
  If a genuine consumer for raw edge manipulation emerges, that's a future
  decision; do not preserve speculatively.
- The knowledge module's embedder wiring and `GraphDbContext` state.

Sweep core for leftovers: anything that existed only to serve the old app
path (e.g. unused re-exports in `storage/mod.rs`) goes.

## Scope

- Knowledge API rewrite, OpenAPI updates, test rewrite on ephemeral
  `Index`, config for the index storage path, dead-code sweep.

## Out of scope

- New product features beyond parity-with-`Index` capability, MCP surface,
  auth, pagination.

## Acceptance

- `./build/scripts/ws-check.sh` and `ws-test.sh` pass.
- App serves add/get/remove/search over records end to end (route tests
  prove it, including a search that exercises relationships and previews).
- No `GraphDbContext` or embedding usage remains in `app` — `Index` is the
  only knowledge dependency.
