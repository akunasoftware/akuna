# Step 09 — App Adoption

Read `00-overview.md` first. Requires steps 07 and 08 (the product ships
the full pipeline — expansion and previews are exercised by acceptance).

## Goal

Rewrite the app's knowledge API on top of `Index`, deleting its bespoke
storage wiring, and give the app a real data-directory story. The plan is
done when the product uses `Index` — the knowledge service is implemented
in `app` on top of core, never in core.

## Context

- `src-crates/app/src/api/knowledge/mod.rs`: routes over raw graph
  nodes/edges (`ApiState` holds a `GraphDbContext`); its search route,
  embedder wiring, and OpenAPI search registrations were already removed
  in step 04.
- `src-crates/app/src/api/knowledge/tests.rs`: route tests against an
  in-memory backend — the pattern to preserve (ephemeral `Index`).
- OpenAPI registrations live in `src-crates/app/src/api/server.rs`
  (utoipa). Core types already carry serde + `ToSchema` (steps 05/06).
- There is NO config module in the app today; the current "pattern" is a
  cwd-relative constant (`GRAPH_DB_NAME = "knowledge"`), and
  `cli/mod.rs` hardcodes `setup_tracing("akuna", ...)` — an existing
  violation of the no-hardcoded-app-name rule. Fix, don't copy.

## Design

**Data directory (new, owner-directed).** Introduce an app-name constant
(single source: a `const` read by tracing, data paths, and anything else —
`env!("CARGO_PKG_NAME")` or a crate-root constant per AGENTS.md's rebrand
rule) and fix the `setup_tracing` call site to use it. Implement data-dir
resolution the way the prior implementation in the **akuna-old** project
does it — consult that repo for the reference, with two owner-directed
deviations:

- akuna-old assumed a single app context; here one data root hosts many
  indexes. The app resolves only the platform data ROOT for the app name;
  `Index` derives its own subpath from `IndexOptions.name` (step 05). The
  app passes `path: Some(<data_root>)`, `name: "knowledge"`.
- akuna-old also had a config dir and config handling — ignore all of it.
  No config dir, no config files, keep it slim; that's not happening any
  time soon.

If akuna-old is not accessible, do NOT block: default to the `directories`
crate's `ProjectDirs` data dir derived from the app-name constant alone
(empty qualifier/organization), note the divergence in the PR, and let the
owner reconcile. Either way this is a new workspace dependency (no
dirs-class crate exists today). This replaces the cwd-relative constant
pattern.

**State and startup.** `ApiState { index: Arc<Index> }`. The `Index` is
built once at server startup (async, loads models, fail-fast with context)
and handed to the router; keep the two-function pattern
(`router()` for production, `router_with_index(index)` as the test seam).

**Routes** (nested under the existing API prefix; follow the app's REST
conventions where they conflict):

- `POST /records` — body: bare JSON array of core `Record`. Upsert; `200`
  echoing the written records (idempotent upsert, not a 201-create).
- `GET /records/{collection}/{id}` — `Index::get`; `200` with the full
  `Record` (content included) or `404`.
- `DELETE /records/{collection}/{id}` — `Index::remove`; `204` always,
  including for records that don't exist (`remove` is idempotent per step
  05 — no 404 probing).
- `GET /records/search` — params: `q` (required, trimmed non-empty else
  `400`), `collections` (comma-separated, absent = all — existing
  convention), `limit` (default from core, bounded app-side like today),
  `filter` (optional URL-encoded JSON of core `MetadataFilter`, exactly
  the serde format pinned in step 05 — no API-local filter dialect).
  Returns `Vec<IndexSearchResult>` with previews populated and expansion
  active.

Response bodies reuse core types — no API-local copies (one shape per
concept). Error mapping: a `From` impl for core's index error into
`ServiceError` (validation/config → 400, missing → 404 in the handler,
rest → 500); malformed `filter` JSON → 400 via the existing serde error
path.

**Delete:**

- Direct graph endpoints (`/graph/nodes`, `/graph/edges`), their handlers,
  param types, label/metadata validators, and `GraphDbContext` state —
  record relationships travel inside record bodies now. (Raw edge access
  returns only if a real consumer emerges; don't preserve speculatively.)
- `GRAPH_DB_NAME` and the per-request `graph()` opener.
- Stale OpenAPI paths/schemas in `server.rs` (`GraphNode`, `GraphEdge`,
  node/edge routes); register the record routes and core schemas instead.

**Sweep core for leftovers:** re-exports, derives, or `From` impls that
existed only for the old app path (e.g. utoipa derives on
`GraphNode`/`GraphEdge` if nothing serializes them anymore, the
`GraphError → ServiceError` impl).

**Tests.** Rewrite on ephemeral `Index`, keeping the `request()` harness
(construction goes async — `router_with_index(Index::new(..).await)` — so
the harness setup changes more than the route diff suggests, and
`server::run` follows). Use slim options for CRUD cases
(`reranking_model: None`, `fulltext: false` — fast, embedder only) and
default options for ONE full-pipeline search case: two linked records,
search surfaces both (expansion, `limit ≥ 2`) with non-null previews
containing the query-relevant text. Model weight: share one slim `Index`
across the CRUD tests (build-once seam, mirroring how core module tests
share model stacks) rather than loading the embedder per test; models
come from the default HF cache like core's tests. Plus: 404 on missing
record GET, 204 on missing record DELETE, upsert-replaces round-trip,
collection scoping, filter param, `400`s (empty `q`, bad `limit`,
malformed `filter`).

## Scope

- App-name constant + data-dir resolution (per akuna-old reference),
  knowledge API rewrite, OpenAPI updates, error mapping, test rewrite,
  core dead-code sweep.

## Out of scope

- New product features beyond `Index` parity, MCP surface, auth,
  pagination.

## Acceptance

- `./build/scripts/ws-check.sh` and `ws-test.sh` pass.
- App serves add/get/remove/search over records end to end; the
  full-pipeline test proves relationships and previews through HTTP.
- No `GraphDbContext`, embedding use, or hardcoded app name remains in
  `app` — `Index` is the only knowledge dependency.
