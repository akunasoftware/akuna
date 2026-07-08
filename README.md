[![License](https://img.shields.io/badge/license-MIT-0f766e?style=for-the-badge)](./LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/akunasoftware/akuna?style=for-the-badge)](https://github.com/akunasoftware/akuna/commits/main)

<h1>
  <img src="./assets/icon-gradient.svg" alt="" width="36" align="left">
  Akuna
</h1>

<p align="center">
  <strong><font color="#00bba7"><em>Knowledge</em></font> is about more than <font color="#155dfc"><em>memory</em></font>.</strong>
</p>

This project aims to service a gap in available semantic & context engineering tooling.

All of the below features, while preserving the following key values:

- **Permissive Core** and **Dependencies**
- **Zero External Runtimes** (no pytorch, onnx etc.)
- **Fully Platform Native**
- **Maximum Performance**

## Key Features

- Batteries Included
  - Rich defaults for painless start
  - Knowledge primitives included

- Sophisticated Tooling
  - Document extraction
  - Structured content parts
  - Entity recognition & reification (WIP)
  - Hardware accelerated vector embedding
  - Graph storage & traversal (WIP)
  - Semantic search
  - Fulltext search
  - ML reranking (WIP)

## [Application](./src-crates/app/)

Command line application, currently implements minimal features. Much more coming here soon.

## [Core Library](./src-crates/core/)

See [`src-crates/core/Cargo.toml`](./src-crates/core/Cargo.toml) for available feature sets.
Use `full` to enable all feature-gated APIs.

`akuna-core` is a single crate with feature-gated modules.

| Module        | Cargo Feature | Description                                             |
| ------------- | ------------- | ------------------------------------------------------- |
| `chunking`    | `chunking`    | Segments text and source files into retrieval chunks.   |
| `detection`   | `detection`   | Classifies raw bytes and files by type.                 |
| `embedding`   | `embedding`   | Loads text embedding models and embeds text batches.    |
| `extraction`  | `extraction`  | Extracts file metadata, text content, and parts.        |
| `index`       | `index`       | Provides an embedded record index.                      |
| `layout`      | `layout`      | Detects document layout blocks in images.               |
| `ocr`         | `ocr`         | Detects and recognizes text in page images.             |
| `reranking`  | `reranking`  | Scores and ranks documents against a query.             |
| `storage`     | `storage`     | Provides graph and vector storage primitives and APIs.  |

Module source lives under [`./src-crates/core/src/`](./src-crates/core/src/).

## Workspace Crates

| Crate         | Path                 | Purpose                                                |
| ------------- | -------------------- | ------------------------------------------------------ |
| `akuna`       | `./src-crates/app/`  | Command-line application and HTTP API.                 |
| `akuna-core`  | `./src-crates/core/` | Knowledge tooling library with feature-gated modules.  |
| `akuna-ffi`   | `./src-crates/ffi/`  | Foreign-language bindings for `akuna-core`.            |

## Documentation

- `akdoc <crate>` — alias to `cargo doc --no-deps --all-features --open -p <crate>`, mirroring docs.rs output for the named crate.
