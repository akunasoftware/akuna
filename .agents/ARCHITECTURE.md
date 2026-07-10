# Architecture

What exists and where, present tense. Hard rules: `PRINCIPLES.md`;
conventions: `CODESTYLE.md`. Planned work gets one pointer here, never a
spec — `.agents/planning/` owns those.

## Workspace

- `src-crates/core` — `akuna-core`, the library.
- `src-crates/app` — the product binary: CLI + axum HTTP API.
- `src-crates/ffi` — UniFFI bindings crate.
- `build/` — nix (devshell, main/oci packages) + `ws-*` scripts. CI runs
  check/test/parity inside `nix develop` plus a 3-arch nix build matrix.

## Core

Feature-gated capabilities. `default = []`; `full` = everything; edges:
`extraction → detection` (+ ~30 tree-sitter grammars), `ocr → layout`,
ML capabilities → private `ml`. app and ffi consume core via
`features = ["full"]` — a feature absent from `full` is invisible to both.

- `detection` — file-type identification: vendored Google Magika
  (upstream Rust + committed weights, converted from ONNX format at build
  time and embedded; nothing fetched at runtime).
- `embedding` — `TextEmbedder`, dense text embeddings.
- `reranking` — `TextReranker`, cross-encoder pair scoring.
- `layout` — `LayoutDetector`, document layout blocks.
- `ocr` — `OcrEngine`, image OCR (composes layout).
- `extraction` — file/bytes → text + structured parts; free functions +
  `ExtractionConfig` (tree-sitter, PDF/office/EPUB/OCR extractors).
- `storage` — graph storage: `GraphDbContext` trait, Grafeo backend,
  `open_context`/`in_memory_context`.
- `ml` (private) — Burn plumbing; device selection is our own GPU probe
  (wgpu Metal/Vulkan, ndarray fallback) over `burn-dispatch`; only wgpu +
  ndarray backends compile.

Checkpoints for embedding/reranking/layout/ocr download from Hugging Face
into a configurable cache at first use; detection is fully embedded.

## App

CLI + HTTP API (`127.0.0.1:9876`, `/api/v1`, utoipa OpenAPI via
`akuna schemas`). Currently opens graph storage at a cwd-relative path and
hardcodes the app name in tracing — known deviations, replacement planned.

## FFI

Modules mirror core capabilities; Python parity tests in
`ffi/tests/python` (uv + pytest via `ws-parity.sh`). Known deviations from
the 1:1 mirror (renames, dropped option fields, the big-stack wrapper) are
core-surface debt per `PRINCIPLES.md`.

## Planned

Indexing — chunking lift, vector storage (LanceDB), `Index` actor, app
adoption: `.agents/planning/indexing/`.
