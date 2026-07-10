# Architecture

What exists and where, present tense. Planned work gets one pointer here,
never a spec.

## Workspace

- `src-crates/core` — the library.
- `src-crates/app` — the product binary: CLI + HTTP API.
- `src-crates/ffi` — multi-language bindings crate.
- `build/` — reproducible packaging, the devshell, and the `ws-*` scripts;
  CI runs check/test/parity in the devshell plus a three-arch package
  build.

## Core

Feature-gated capabilities. Empty default feature set; `full` = everything;
app and ffi consume `full`. Edges: extraction → detection (+ ~30 parser
grammars), ocr → layout, ML capabilities → the private ML plumbing module.

- `detection` — file-type identification; vendored model with committed
  weights, converted at build time and embedded — nothing fetched at
  runtime.
- `embedding` — dense text embeddings.
- `reranking` — cross-encoder pair scoring.
- `layout` — document layout blocks.
- `ocr` — image OCR (composes layout).
- `extraction` — file/bytes → text + structured parts (code parsing plus
  PDF/office/EPUB/OCR extractors); free functions + an options struct.
- `storage` — graph storage: a context trait over an embedded graph
  engine, on-disk and in-memory constructors. Core-only; not exposed over
  FFI.
- `ml` (private) — ML plumbing; device selection via our own GPU probe
  over the framework's dispatch layer; GPU and CPU backends only.

Checkpoints for the four model capabilities download from the model host
into a configurable cache at first use; detection is fully embedded.

## App

CLI + loopback-only HTTP API with a versioned path prefix and a generated
API schema (schemas subcommand). Known deviations: graph storage opens at
a cwd-relative path; the app name is hardcoded in logging. Replacement
planned.

## FFI

Modules mirror core's model and extraction capabilities (storage is
core-only). Generator-based bindings; parity tests live beside the
bindings crate and run via `ws-parity.sh`. Known mirror deviations
(renames, dropped option fields, the big-stack wrapper) are core-surface
debt.

## Planned

Indexing — chunking lift, vector storage, index actor, app adoption:
`.agents/planning/indexing/`.
