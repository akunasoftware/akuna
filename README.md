[![License](https://img.shields.io/badge/license-MIT-0f766e?style=for-the-badge)](./LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/akunasoftware/akuna?style=for-the-badge)](https://github.com/akunasoftware/akuna/commits/main)

<h1>
  <img src="./assets/icon-gradient.svg" alt="" width="36" align="left">
  Akuna
</h1>

<p align="center">
  <strong><font color="#00bba7"><em>Knowledge</em></font> is about more than <font color="#155dfc"><em>memory</em></font>.</strong>
</p>

This project aims to service a gap in available context engineering tooling.

Fully featured, while preserving the following key values:

- **Permissive**
  - Open source licensed core
  - All dependencies permissive FOSS
- **Fully Platform Native**
  - AI/ML features can run locally
  - No external runtimes (no pytorch, onnx etc.)
  - Support all common operating systems & device architectures
- **Resource Conscious**
  - Tiny bundle size (given the feature set)
  - Respectful of memory & CPU footprint
- **Batteries Included**
  - Opinionated primitive concepts included
  - Sensible defaults everywhere, for painless start
  - Simple top level interfaces, no advanced familiarity required

## Workspace Crates

| Crate        | Path                                       | Purpose                                                                       |
| ------------ | ------------------------------------------ | ----------------------------------------------------------------------------- |
| `akuna`      | [`./src-crates/app/`](./src-crates/app/)   | Main command line application, currently minimal. Much more coming here soon. |
| `akuna-core` | [`./src-crates/core/`](./src-crates/core/) | Core rust library. Knowledge tooling library with feature-gated modules.      |
| `akuna-ffi`  | [`./src-crates/ffi/`](./src-crates/ffi/)   | Foreign-language bindings for `akuna-core` (for use in python, js etc.)       |

## Core Library Features

`akuna-core` combines independently feature-gated capabilities.

Use `full` to enable all feature-gated APIs.

See [`src-crates/core/Cargo.toml`](./src-crates/core/Cargo.toml) for available feature sets.

| Module                                            | Cargo Feature | Description                                                                                  |
| ------------------------------------------------- | ------------- | -------------------------------------------------------------------------------------------- |
| [`detection`](./src-crates/core/src/detection/)   | `detection`   | Infers file types from raw bytes and files with Magika.                                      |
| [`embedding`](./src-crates/core/src/embedding/)   | `embedding`   | Creates hardware-accelerated vector embeddings for text batches with multiple model choices. |
| [`extraction`](./src-crates/core/src/extraction/) | `extraction`  | Extracts structured file metadata, text content, and parts.                                  |
| [`ocr`](./src-crates/core/src/ocr/)               | `ocr`         | Detects and recognizes text in images.                                                       |
| [`reranking`](./src-crates/core/src/reranking/)   | `reranking`   | Uses ML models to score and rank documents against a query.                                  |

### In Progress & Coming Soon

- Vision language model inference
- Syntax-tree-aware extraction
- Entity recognition & reification
- Simple-to-use advanced-capability search & retrieval methods
- Abstracted storage standards for graph & vector databases
- Live index building for rapid retrieval
- Live 'alpha' building, to autonomously boost contents' meaning & usefulness
