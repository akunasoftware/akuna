# Indexing Plan

## Goal

Add an `Index` abstraction in `core` and FFI.

`Index` should hide storage and search details. Users should add documents,
remove documents, and search. They should not manage vector stores, graph
stores, embeddings, reranking, or OCR directly for common indexing work.

## Shape

`Index` is an actor type.

Options use the normal flat shape:

- `Index`
- `IndexOptions`

No builder pattern.

Default construction should work.

## Opening

An index can be created with a storage path or without one.

- `None`: in-memory or temporary storage.
- `Some(path)`: persistent storage rooted at that path.

Future use: when scanning exists, a path may also let the index discover and
index files automatically. Do not build scanning yet.

## Collections

An index owns many collections.

Collections isolate documents and chunks inside one index.

Search default:

- no collection filter means search all collections.
- one collection means search only that collection.
- many collections means search across those collections.

## Documents

Index should support a ChromaDB-like flow:

- add documents
- update documents
- remove documents
- search documents
- filter by metadata

First input shape can be plain documents with text and metadata. Extraction can
feed this later.

Likely types:

- `IndexDocument`
- `IndexChunk`
- `IndexSearchQuery`
- `IndexSearchResult`
- `IndexMetadataFilter`

## Storage

`Index` is an abstraction over storage backends.

Storage layer needs vector storage.

Use LanceDB for vector storage.

Vector storage stores chunks:

- chunk id
- collection
- document id
- text
- embedding
- metadata

Graph storage stores things, not strings:

- documents
- entities
- relationships

Graph storage should not store chunks.

## Search

Search starts with vector retrieval over chunks.

Metadata filtering is required.

Reranking is configurable and can be applied after retrieval.

Result should include enough data to trace back:

- collection
- document id
- chunk id
- text
- metadata
- score

## Configuration

`IndexOptions` should configure:

- storage path
- vector storage mechanism
- graph storage mechanism
- embedding model
- reranker model
- OCR model

Maybe later:

- chunking options
- search limits
- rerank limits
- metadata filter behavior

Do not configure detection model for now. Detection belongs to extraction and
path handling.

## Core

Core owns `Index`.

Core should also own vector storage traits and LanceDB backend.

Keep the public API small. Add only what `Index` needs first.

## FFI

FFI exposes `Index` with the same names and shapes as core.

FFI should stay dumb:

- annotations
- conversions
- no indexing behavior

Path methods should use Python naming rules:

- Rust path variant: `_file`
- Python FFI path variant: `_path`

## First Milestone

No extraction.

No scanning.

No graph entity recognition.

Build only:

- `Index`
- `IndexOptions`
- collection-aware add/update/remove
- vector chunk storage with LanceDB
- metadata filtering
- search across all, one, or many collections
- FFI mirror

## Open Questions

- Exact metadata filter shape: simple equality first, or boolean operators now?
- Should collection creation be explicit, or created on first document add?
- Should update replace all chunks for a document?
- Should deletes remove by document id, collection plus document id, or both?
