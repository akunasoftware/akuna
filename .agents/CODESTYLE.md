# Code Style

Code conventions. Deviations need owner sign-off.

## Naming

- Actor types are agent nouns saying what they do, e.g. `TextEmbedder`,
  `FileTypeDetector`.
- Options structs are `<Actor>Options`; model enums stay domain-named.
- Byte/path method pairs are `<verb>_bytes` + `<verb>_file` in core; FFI
  path variants are `<verb>_path`.

## Options

- Flat plain-field structs, strong types, enums for choices. No builders.
- Default construction always works; manual `Default` when derives would
  be wrong. Model-actor options carry an optional cache-dir override.

## Errors

- One typed error enum per capability, exported from the feature root from
  a dedicated error module, context + source preserved. Convert
  engine/library errors at the boundary — never leak them through domain
  APIs.
- Missing-item reads return `Option`, not errors. No `unwrap`; `expect`
  only for provable invariants, reason in the message.

## Modules

- Capability layout: `mod.rs` = types + actor + curated `pub use` (never
  wildcard); model impls under `models/`, storage engines under
  `backend/`; tests as `#[cfg(test)] mod tests;` + sibling `tests.rs`.
- Module doc: one-sentence `//!` summary + a runnable example doctest.
- Internals private by default; narrow visibility only where truly
  imported. Vendored, model, and generated modules stay private.
- Every capability is a feature flag, aggregated in `full`. Verify
  standalone feature builds yourself.

## Imports

- Blank-line groups: std / external / crate. Prefer `crate::` absolute
  paths over `super::`. No inline paths in signatures.

## ML actors

- Construction: public async `new(options)` delegating to a crate-private
  `new_on(device, options)`; model enum variants doc their upstream
  checkpoint repo, mapped via a private repo-id helper.

## FFI

- Per-module string-only error enum + conversion helper. Construction is a
  free async factory taking optional options, defaulting to core's
  defaults. Stack-heavy inference goes through the shared big-stack
  wrapper; light inference calls core directly.

## Tests

- Names short, implicit from module path; consistent wording so names sort
  into readable groups.
- Fixtures download from the hosted test corpus via the shared testkit;
  model-heavy tests use its big-stack runner.

## Dependencies

- Shared deps live at workspace level; comment the rationale for
  non-obvious choices. Manage versions with the package-manager CLI, never
  by hand.

## Docs

- Docstrings on all public items — purpose per `PRINCIPLES.md`. The lint
  config and check scripts enforce the rest; don't fight them.

## Determinism

- Same input + options → same output wherever possible. Tie-break
  explicitly; never depend on engine return order.
