# Code Style

Code conventions. Deviations need owner sign-off; where legacy code
differs, these rules win on next touch. Hard rules live in `PRINCIPLES.md`.

## Naming

- Actor types are agent nouns saying what they do: `TextEmbedder`,
  `TextReranker`, `LayoutDetector`, `OcrEngine`, `FileTypeDetector`.
- Options structs are `<Actor>Options`; model enums stay domain-named
  (`EmbeddingModel`, `OcrDetectionModel`).
- Byte/path method pairs are `<verb>_bytes` + `<verb>_file` in Rust; FFI
  path variants are `<verb>_path`.

## Options

- Flat plain-field structs, strong types, enums for choices. No builders.
- Default construction always works; manual `Default` impl when derives
  would be wrong. Model-actor options carry `cache_dir: Option<PathBuf>`.

## Errors

- One typed `thiserror` enum per capability, exported from the feature
  root, context + `source` preserved. Convert engine/library errors at the
  boundary — never leak them through domain APIs.
- embedding/reranking's `anyhow` returns are legacy — migrate on next
  touch.
- Missing-item reads return `Option`, not errors. No `unwrap`
  (lint-denied); `expect` only for provable invariants, reason in the
  message.

## Modules

- Capability layout: `mod.rs` (types + actor + curated `pub use`, never
  wildcard), `models/` for ML model impls (`backend/` for storage
  engines), `error.rs` for the enum, `#[cfg(test)] mod tests;` + sibling
  `tests.rs`.
- Module doc: one-sentence `//!` summary + runnable ```rust,no_run
  example doctest.
- Internals private by default; `pub(crate)`/`pub(in ...)` only when truly
  imported elsewhere. Vendored, model, and generated modules stay private.
- Every capability is a cargo feature, aggregated in `full`; app and ffi
  consume `full`. CI runs `--all-features` only — verify standalone builds
  yourself: `cargo check -p <core crate> --no-default-features --features <f>`.

## Imports

- Blank-line groups: std / external / crate. Prefer `crate::` absolute
  paths over `super::`. No inline paths in signatures.

## ML actors

- Construction: `pub async fn new(options)` delegating to
  `pub(crate) new_on(device, options)` for device injection; model enum
  variants doc their HF repo, mapped via a private `repo_id()`.

## FFI

- Per-module string-only error enum + `to_error` helper; free async
  factory `load_<actor>(Option<Options>)` defaulting via a `core_options`
  helper so binding defaults stay aligned with core; sync inference runs
  through the shared big-stack wrapper (`stack.rs`).

## Tests

- Names short, implicit from module path; consistent wording so names sort
  into readable groups.
- Fixtures download from the HF test corpus via `testkit`; model-heavy
  tests run via `run_with_model_stack`.

## Dependencies

- Workspace-level deps (`workspace = true`), exact version pins, one
  rationale comment each. Manage with the cargo CLI.

## Docs & lints

- Docstrings on all public items (`missing_docs` denied); purpose not
  mechanics per `PRINCIPLES.md`.
- Workspace denies `warnings`, `clippy::all`, `clippy::unwrap_used`,
  `missing_docs`; ws-check adds `cargo doc`, `cargo deny`, `cargo machete`.

## Determinism

- Same input + options → same output wherever it can possibly hold.
  Tie-break explicitly; never depend on engine return order.
