# Indexing Plan — Overview

Read this file before implementing any numbered step. It holds the contracts
every step must honor. Each numbered file is one PR-sized step, implemented in
order. `AGENTS.md` and `.agents/principles.md` are binding on all steps.

## Vision

`Index` is an embedded retrieval store over **Records**. Consumers add
records, remove records, and search — they never manage vector stores, graph
stores, embeddings, chunking, reranking, or OCR directly.

- Rust and Python ergonomics matter equally.
- `Index` must be fully usable with no filesystem involvement: consumers may
  embed it inside an outer software stack and feed it records directly.
- Later (not in this plan): given a storage path, `Index` will be able to
  scan and index files itself, and NER will populate graph entities.

## Concepts

- **Index** — the central actor in core. Owns the storage engines and ML
  models. Named `Index` by explicit owner exemption from the agent-noun
  convention: it is a stateful store, and `Indexer` would wrongly suggest a
  stateless transformer.
- **Record** — the unit of content. Has id, collection, title, content,
  metadata, and optional relationships to other records.
- **Collection** — isolates records inside one index. Created implicitly on
  first add. Record ids are unique within a collection.
- **Chunk** — internal implementation detail for vector retrieval. Chunks
  NEVER appear in any public API, result, or FFI surface.
- **Relationship** — typed edge between records. Caller-supplied for now.
  Extraction will soon use this to link records by file path; NER-extracted
  entities come later as new node kinds.

## Engines

Two storage layers, strict division of labor:

- **Vector layer** (`storage/vector`, trait + LanceDB backend): owns ALL
  searching — dense chunk vectors, BM25 over chunk text, dense title
  vectors, BM25 over titles — AND is the authoritative store for record
  content: each record has a record row (title, title embedding, full
  content, metadata) alongside its chunk rows.
- **Graph layer** (`storage/graph`, existing Grafeo backend): pure
  structure — records as nodes (title, collection, metadata; NO content)
  and relationships as edges, answering traversal queries. The graph layer
  does NO string or vector search — search is removed from its API in step
  04 so any graph backend can satisfy the trait.

**Metadata rule (binding):** every engine stores the same record metadata and
collection so metadata/collection filtering behaves consistently regardless
of which engine produced a candidate. (Vector engines filter at query time
in-engine; graph traversal results are filtered by the pipeline using the
same `MetadataFilter` semantics.)

**Async rule:** the vector layer and `Index` are async end to end (LanceDB's
Rust API is async; the app server is async). The graph trait stays sync
(Grafeo is sync).

## Search pipeline

1. **Candidates** — dense chunk search + dense title search (always) and
   BM25 chunk search + BM25 title search (when `fulltext` is enabled), fused
   by record with Reciprocal Rank Fusion. Collection and metadata filters
   are pushed into every retrieval call.
2. **First rerank** — each candidate record's evidence set is its retrieved
   chunk texts PLUS its title when it had a title hit; rerank evidence
   against the query, roll up to records (max evidence score). The winning
   evidence travels with the candidate, kind-tagged, through the rest of
   the pipeline (expansion and previews branch on it).
3. **Graph expansion** — traverse relationships from candidate records to
   widen the packet; expanded records are hydrated from the vector layer.
4. **Final aggregation** — rerank the expanded set on bounded
   representative text (skipped when expansion added nothing; see step 07).
5. **Results** — record-level only: record id, collection, title, metadata,
   score, `preview: Option<String>`. Preview is a semantically-centered
   truncated string (step 08); `None` until then. Full content is read
   separately via `get`.

## Configuration

- Flat `IndexOptions`, strongly typed, enums for choices, sensible defaults —
  default construction always works. No builders.
- Dense retrieval is always on. `fulltext` gates both BM25 functions (their
  indexes are only created when enabled); `graph` gates graph storage and
  expansion entirely. Disabled functions cost nothing.
- Reranking is ON by default.
- Every index has a `name`; one data root hosts many indexes, each stored
  under `<root>/<name>`. `path: None` → ephemeral storage (temp root,
  identical layout). `path: Some(dir)` → persistent storage at
  `<dir>/<name>`, with a manifest guarding config compatibility on
  reopen.

## Conventions

- API style per `AGENTS.md`: actor nouns, `<Actor>Options`, `_bytes`/`_file`
  in Rust and `_path` in Python FFI, flat options.
- Docstrings state purpose, never mechanics; rationale lives in `//` code
  comments (`.agents/principles.md`). The type sketches in these step files
  annotate semantics for the implementer — do not copy sketch comments into
  `///` docs verbatim.
- The FFI mirror covers `Index`'s public consumer surface 1:1 (types,
  names, shapes). Internal storage traits and the chunker are core-only and
  never cross FFI. FFI work starts in step 05, where `Index` first exists;
  UniFFI cannot generate async primary constructors for Python, so
  construction is exposed as a free async factory function per the existing
  ffi convention (`load_text_embedder`).
- Validate with `./build/scripts/ws-check.sh` and `ws-test.sh`; FFI steps
  also run `ws-parity.sh`.

## Steps

| Step | Scope | Depends on |
| --- | --- | --- |
| 01-vector-storage | `storage/vector` async trait + LanceDB backend; chunk + record rows; dense search; in-engine metadata filtering; batch hydration | — |
| 02-fulltext-search | BM25 over chunk text and titles in the same backend; index-config on open | 01 |
| 03-chunking | Lift extraction's segmentation into a `chunking` namespace + packer + `ChunkingOptions` | — |
| 04-record-graph | Remove search from the graph API; add traversal; slim grafeo features | — |
| 05-index-core | `Index`/`IndexOptions`, records CRUD, manifest, FFI mirror | 01–04 |
| 06-search-pipeline | Candidates → fusion → rerank → record results; expansion seam | 05 |
| 07-graph-expansion | Relationship traversal stage + final aggregation | 06 |
| 08-preview | Semantically-centered preview builder | 07 |
| 09-app-adoption | App config/data dirs; knowledge API rewritten on `Index` | 07 + 08 |
