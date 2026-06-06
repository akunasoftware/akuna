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

### Module Exports

- `lib.rs`: expose only feature modules with `pub mod <feature>`.
- Feature `mod.rs`: keep internals private by default.
- Public API: explicit `pub use`; never wildcard-export `::*`.
- Public errors: typed per feature, exported from feature root.
- Public call points: root-level or one deliberate domain module; do not mix casually.
- Nested modules public only when stable API, e.g. `extraction::document`.
- Vendor, model, generated modules stay private.
- Use `pub(crate)` only when another module truly imports it.

### Tests

- Test names: short, strongly implicit from module path.
  - Do not restate namespace already clear from file/module path.
  - Keep wording/order consistent so names sort into readable groups.

## Docs

- Be concise. Optimize quick reading. Cut cruft.
