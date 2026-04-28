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
  - Content chunking
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
Use `full` to enable all feature-gated APIs

| Namespace                                         | Cargo Features                     | Description                                          |
| ------------------------------------------------- | ---------------------------------- | ---------------------------------------------------- |
| [`Extraction`](./src-crates/core/src/extraction/) | `extraction`                       | Extracts file metadata, text content, and chunks.    |
| [`Chunking`](./src-crates/core/src/chunking/)     | `chunking`, `chunking-tree-sitter` | Splits text using configured delimiters and size.    |
| [`Embedding`](./src-crates/core/src/embedding/)   | `embedding`                        | Loads text embedding models and embeds text batches. |
| [`Graph`](./src-crates/core/src/graph/)           | `graph`                            | Provides graph primitives, types, and storage APIs.  |
| [`Types`](./src-crates/core/src/types/)           |                                    | Shared result, error, and configuration types.       |

## Additional Crates

| Crate                                                         | Purpose                                                                                                             |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| [`burn-magika`](https://github.com/akunasoftware/burn-magika) | Rust native [Magika](https://github.com/google/magika) file type inference built on [Rust Burn](https://burn.dev/). |
| [`burn-embed`](https://github.com/akunasoftware/burn-embed)   | Rust native text embedding models built on [Rust Burn](https://burn.dev/).                                          |
