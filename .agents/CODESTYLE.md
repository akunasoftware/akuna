# Code Style

Code conventions. Follow everywhere; deviations need owner sign-off.
Hard rules and purpose live in `PRINCIPLES.md`.

## Naming

- Actor types are agent nouns saying what they do: `TextEmbedder`,
  `TextReranker`, `LayoutDetector`, `OcrEngine`, `FileTypeDetector`.
  (`Index` is an explicit owner exemption — a stateful store, not a
  transformer.)
- Options structs are `<Actor>Options`.
- Model enums stay domain-named: `EmbeddingModel`, `OcrDetectionModel`.
- Byte/path method pairs are `<verb>_bytes` and `<verb>_file` in Rust;
  Python FFI uses `<verb>_path` for path variants.

## Options

- Flat plain-field structs, strong types, enums for choices. No builders.
- Default construction always works; write a manual `Default` impl when
  derived defaults would be wrong (zeroes, `None` where a model belongs).
- Configure only what genuinely warrants it; configurability is
  complexity.

## Errors

- Storage layers: `thiserror` enums with engine-tagged variants that
  preserve `source` (`GraphError` is the template).
- Model actors and composites: `anyhow::Result` with context; messages
  that tests assert on must be distinct and stable.
- Missing-item reads return `Option`, not errors. Deletes of absent
  items: match the layer's documented convention and test it.

## Async

- Async only where the underlying work is async (LanceDB, model
  loading/downloads, `Index`). Sync inference stays sync; never call it
  inline on an async path — wrap (e.g. `spawn_blocking`).
- Traits needing `Box<dyn>` object safety with async methods use the
  `async-trait` crate.

## Modules

- Capability layout: `mod.rs` (domain types + trait + constructors),
  `backend/` (concrete impls), `error.rs`, sibling `tests.rs` declared
  via `#[cfg(test)] mod tests;` (see `embedding/`, `reranking/`).
- Every capability sits behind a cargo feature, aggregated in `full`.
  CI runs `--all-features` only — verify standalone feature builds
  yourself: `cargo check -p akuna-core --no-default-features --features
  <feature>`.
- Keep the public surface minimal; internal helpers are `pub(crate)` or
  tighter. A `pub fn` must not return a crate-private type.

## Documentation

- Docstrings on all public items — the workspace denies `missing_docs`.
- Purpose, never mechanics (see `PRINCIPLES.md`); rationale that must
  survive lives in `//` code comments at the decision site.
- Comment density matches the surrounding code; no narrating the obvious.

## Lints & checks

- Workspace denies: `warnings`, `clippy::all`, `clippy::unwrap_used`,
  `missing_docs`. Plus `cargo doc`, `cargo deny`, `cargo machete` in
  `ws-check.sh`.
- Validate with `./build/scripts/ws-check.sh` (fast) and `ws-test.sh`
  (exhaustive); FFI changes also run `ws-parity.sh`.

## Determinism

- Same input + options → same output, everywhere it can possibly hold
  (chunking, search ordering, previews). Tie-break explicitly; never
  depend on engine return order.
