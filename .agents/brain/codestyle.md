# Code Style

## Rust

### Errors

- Never use `unwrap`. Propagate errors well.
- Match project error surfaces.
  - Core/domain: typed `thiserror` enums with useful context + `source`.
    Find current examples before adding new patterns.
  - Convert engine/library errors at boundaries; do not leak them into domain APIs.

### Dependencies

- Manage deps with `cargo` CLI.
- Never hardcode versions in package config; version memory goes stale.
- `cargo` gets latest deps.

### Imports

- No inline imports in function signatures, e.g. `crate::appconfig::AppConfig`.
- Prefer `crate::` absolute imports over `super::` cross-module imports.

### Tests

- Test names: short, strongly implicit from module path.
  - Do not restate namespace already clear from file/module path.
  - Keep wording/order consistent so names sort into readable groups.

## Docs

- Be concise. Optimize quick reading. Cut cruft.
