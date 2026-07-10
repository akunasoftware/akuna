# Architecture

System structure: what lives where and why. Purpose and hard rules live
in `PRINCIPLES.md`; code conventions in `CODESTYLE.md`. Items marked
*planned* are specified in `.agents/planning/` and not yet built.

## Crates

- `src-crates/core` — the library. All sophisticated tooling, embedded.
- `src-crates/app` — the product: CLI + HTTP API (axum) on top of core;
  MCP next. The knowledge service is implemented here, never in core.
- `src-crates/ffi` — dumb UniFFI wrapper exposing core 1:1 to other
  languages. Annotations and type conversions only; async construction
  is a free async factory function (`load_*`) because UniFFI cannot emit
  async primary constructors. Python parity tests in
  `src-crates/ffi/tests/python/` (run via `ws-parity.sh`).

## Core capabilities

Each is a feature-gated module with a small actor-based API:

- `detection` — file-type detection (Magika-class model).
- `embedding` — dense text embeddings (`TextEmbedder`).
- `reranking` — cross-encoder scoring (`TextReranker`).
- `ocr` — image OCR (`OcrEngine`).
- `layout` — document layout detection (`LayoutDetector`).
- `extraction` — file → text + structured parts pipeline (tree-sitter,
  PDF/office/EPUB/OCR extractors).
- `chunking` (*planned*, step 03) — segmentation lifted out of
  extraction (code strategy via tree-sitter, prose strategy) plus the
  retrieval chunk packer. Extraction consumes it; so does `Index`.
- `storage` — persistence layers (below).
- `index` (*planned*, step 05) — the `Index` actor composing storage,
  chunking, embedding, and reranking behind one Records API.
- `ml` (private) — shared Burn model plumbing.

## ML stack

Burn is the sole ML backend; inference runs embedded with hardware
acceleration dispatched automatically (`burn_dispatch`). No external
runtimes — no ONNX Runtime, no PyTorch, nothing outside our own.
Checkpoints download from Hugging Face into a configurable cache.

## Storage engines

Strict division of labor (*target state per the indexing plan*):

- **Vector layer** (`storage/vector`, *planned*, trait +
  LanceDB backend): owns ALL searching — dense and BM25, over chunk text
  and record titles — and is the authoritative store for record content
  (per-record row: title, title embedding, content, metadata; chunk rows
  alongside).
- **Graph layer** (`storage/graph`, Grafeo backend): pure structure —
  nodes, edges, traversal. No string or vector search (removal from the
  current API is plan step 04).
- Both engines store the same record metadata + collection so filtering
  is consistent everywhere (`Metadata`/`MetadataFilter` at the storage
  root).
- No locking anywhere — no lockfiles, no single-instance guards, no
  write mutexes; engine-native concurrency behavior stands (binding; see
  the indexing plan overview).

## Index (planned)

`Index` is an embedded retrieval store over **Records** (id, collection,
title, content, metadata, relationships). Chunks are internal to vector
retrieval and never appear in any public surface. Search pipeline:
candidate retrieval (dense + BM25, chunks + titles) → RRF fusion → chunk
rerank → graph expansion via relationships → final aggregation →
record-level results with optional previews. One data root hosts many
indexes; each `Index` has a `name` driving its `<root>/<name>` subpath,
with a manifest guarding config compatibility. Full specification:
`.agents/planning/indexing/`.

## App

Axum HTTP API + CLI. Holds an `Arc<Index>` built once at startup
(*planned*, step 09); resolves only the platform data root from the
app-name constant — index subpaths belong to core. No config dir or
config files. Never hardcode the app name.
