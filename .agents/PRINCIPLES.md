# Principles

Hard rules. Binding on all work. Add sparingly, with owner sign-off.

## Purpose

The Rust library is the backing toolset for knowledge work tools:
sophisticated functionality behind the simplest possible API surface.

Why it exists: building this in Python meant poor performance and stitching
together a confusing ecosystem of packages and services — nothing did the
whole job in one place. Rust alternatives covered the features but leaned on
external runtimes that made distribution painful. So this is the package
that didn't exist: the whole toolset, embedded, distributable anywhere,
simple to use from any language — for us and for anyone who wants the same.

Three crates, three roles:

- `src-crates/core` — the library. All sophisticated tooling, embedded. It
  exists to give the product its intelligence.
- `src-crates/app` — the product. CLI and HTTP API today; MCP next, and
  eventually a distributed cloud service. The knowledge service is
  implemented here, on top of core — never in core.
- `src-crates/ffi` — the dumb wrapper exposing core to other languages.

- Rust exists here to embed anywhere: one native, memory-safe, compiled
  artifact importable into any language or software stack a customer
  chooses. This covers everything the library does, ML included.
- Burn is the sole ML backend: inference runs embedded and leverages host
  hardware acceleration automatically whenever the platform offers it —
  consuming developers never think about it.
- **No external runtimes.** Nothing may depend on ONNX Runtime, PyTorch, or
  any runtime outside our own embedded one — not for inference, not for
  storage, not for anything.
- Every capability is configurable through a flat options struct with
  sensible defaults — default construction always works.
- Configure only what genuinely warrants it; configurability is complexity.
- One naming and defaulting convention everywhere (AGENTS.md API Style).
- No builder patterns; static options structs are sufficient.

## Architecture

- **Simplest surface.** Expose the minimum API that serves the purpose;
  every public item earns its place.
- **FFI is a dumb wrapper.** Export annotations and type conversion only;
  all implementation lives in core. If a binding needs behavior, add a core
  API — never adapter code.
- **Bindings mirror core 1:1.** Same operations, names, and shapes in every
  language. A binding that must reshape core is a core-surface bug.
- **One shape per concept.** Each piece of information has exactly one
  type, reused everywhere it appears. No partial copies — consumers take
  the full shape and ignore what they don't need.

## Documentation

- **Purpose, never mechanics.** Docstrings and docs say what a thing is for
  — concrete about intent, abstract about internals. No implementation
  details, behavior specifics, or technical terminology: those change, and
  stale references are doc debt.
- Rationale that must survive lives in `//` code comments, not docs.
