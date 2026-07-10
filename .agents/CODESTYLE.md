# Code Style

Conventions. Deviations need owner sign-off.

## Naming

- Actors = agent nouns: `TextEmbedder`, `FileTypeDetector`.
- Options = `<Actor>Options`. Model enums domain-named.
- Pairs, where file input exists: `<verb>_bytes` + `<verb>_file` in core;
  `<verb>_path` in FFI.

## Options

- Flat fields, strong types, enums. No builders. `Default` always works.
- Model-actor options carry an optional cache-dir override.

## Errors

- One typed error enum per capability in a dedicated module, exported from
  the feature root, context + source kept.
- Missing reads → `Option`. No `unwrap`; `expect` only for provable
  invariants, reason in message.

## Modules

- `mod.rs` = types + actor + curated `pub use` (no wildcards); model impls
  in `models/`, storage engines in `backend/`; tests =
  `#[cfg(test)] mod tests;` + sibling `tests.rs`.
- Module doc: one-line `//!` + runnable doctest.
- Private by default. Vendored/model/generated modules stay private.
- Capability = feature flag, aggregated in `full`. Verify standalone
  feature builds.

## Imports

- Groups: std / external / crate. `crate::` over `super::`. No inline
  paths in signatures.

## ML actors

- Public async `new(options)` → crate-private `new_on(device, options)`.
- Model enum variants doc their upstream checkpoint repo; private repo-id
  helper.

## FFI

- Per-module string-only error enum + converter. Free async factory taking
  optional options, defaulting to core's. Inference that can overflow
  default thread stacks (measure when unsure) goes via the shared
  big-stack wrapper; light calls core directly.

## Tests

- Names short, implicit from module path, consistent wording.
- Fixtures via the shared testkit from the hosted corpus; model-heavy
  tests use its big-stack runner.

## Dependencies

- Shared deps at workspace level; rationale comments for non-obvious
  choices. Package-manager CLI only; never hand-edit versions.

## Docs

- Docstrings on all public items — purpose per `PRINCIPLES.md`. Lints and
  check scripts enforce the rest.

## Determinism

- Same input + options → same output. Tie-break explicitly; never trust
  engine order.
