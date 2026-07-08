# Indexing Plan — Overview

Read this file before implementing any numbered step. It holds the contracts
every step must honor. Each numbered file is one PR-sized step, implemented in
order. `AGENTS.md` and `.agents/principles.md` are binding on all steps.

## Vision

`Index` is an embedded retrieval store over **Records**. Consumers add
records, remove records, and search — they never manage vector stores, graph
stores, embeddings, chunking, reranking, or OCR directly.

- Rust and Python ergonomics matter equally. The FFI surface mirrors core 1:1.
- `Index` must be fully usable with no filesystem involvement: consumers may
  embed it inside an outer software stack and feed it records directly.
- Later (not in this plan): given a storage path, `Index` will be able to
  scan and index files itself, and NER will populate graph entities.

## Concepts

- **Index** — actor type in core. Owns the storage engines and ML models.
- **Record** — the unit of content. Has id, collection, title, content,
  metadata, and optional relationships to other records.
- **Collection** — isolates records inside one index. Created implicitly on
  first add.
- **Chunk** — internal implementation detail for vector retrieval. Chunks
  NEVER appear in any public API, result, or FFI surface.
- **Relationship** — typed edge between records. Caller-supplied for now.
  Extraction will soon use this to link records by file path; NER-extracted
  entities come later as new node kinds.

## Engines

Two storage layers, strict division of labor:

- **Vector layer** (`storage/vector`, trait + LanceDB backend): owns ALL
  searching — dense chunk vectors, BM25 over chunk text, dense title
  vectors, BM25 over titles.
- **Graph layer** (`storage/graph`, existing Grafeo backend): stores records
  as nodes (full content in the node payload) and relationships as edges,
  and answers traversal queries. The graph layer does NO string or vector
  search — search capabilities are removed from its API in step 04 so any
  graph backend can satisfy the trait.

**Metadata rule (binding):** every engine stores the same record metadata and
collection so metadata/collection filtering behaves identically regardless of
which engine produced a candidate.

## Search pipeline

1. **Candidates** — dense chunk search + BM25 chunk search + dense title
   search + BM25 title search (each only if its retrieval function is
   enabled), fused into one candidate set. Collection and metadata filters
   apply at retrieval time in every engine.
2. **First rerank** — rerank candidate chunk/title texts, roll scores up to
   records.
3. **Graph expansion** — traverse relationships from candidate records to
   widen the packet.
4. **Final aggregation** — score the expanded record set (design decision in
   step 07) and produce the final ranking.
5. **Results** — record-level only: id, collection, title, metadata, score,
   `preview: Option<String>`. Preview is a semantically-centered truncated
   string (step 08); `None` until then. Full content is read separately via
   `get`.

## Configuration

- Flat `IndexOptions`, strongly typed, enums for choices, sensible defaults —
  default construction always works. No builders.
- Retrieval functions are configured at init; only enabled functions get
  their storage/indexes created and only they contribute candidates.
- Reranking is ON by default.
- `path: None` → ephemeral storage. `path: Some(dir)` → persistent storage
  rooted there, with a manifest guarding config compatibility on reopen.

## Conventions

- API style per `AGENTS.md`: actor nouns, `<Actor>Options`, `_bytes`/`_file`
  in Rust and `_path` in Python FFI, flat options.
- FFI is a dumb mirror; it gains `Index` in step 05 and stays 1:1 in every
  step after. Steps 01–04 are core-internal surface only.
- Validate with `./build/scripts/ws-check.sh` and `ws-test.sh`; FFI steps
  also run `ws-parity.sh`.

## Steps

| Step | Scope | Depends on |
| --- | --- | --- |
| 01-vector-storage | `storage/vector` trait + LanceDB backend; chunk/title tables; dense search; metadata filtering | — |
| 02-fulltext-search | BM25 over chunk text and titles in the same backend | 01 |
| 03-chunking | Internal chunker + public `ChunkingOptions` | — |
| 04-record-graph | Remove search from the graph API; add traversal; slim grafeo features | — |
| 05-index-core | `Index`/`IndexOptions`, records CRUD, manifest, FFI mirror | 01–04 |
| 06-search-pipeline | Candidates → fusion → rerank → record results | 05 |
| 07-graph-expansion | Relationship traversal stage + final aggregation | 06 |
| 08-preview | Semantically-centered preview builder | 06 |
| 09-app-adoption | App knowledge API rewritten on `Index` | 06 (07/08 preferred) |
