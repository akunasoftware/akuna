# Architecture

Demanded shape. Code aligns to this.

## Purpose

One embedded package: the whole knowledge-tooling toolset, usable from any
language, anywhere.

## Split

- `src-crates/core` — the library. All capabilities, all intelligence.
- `src-crates/app` — the product. CLI + HTTP API on the library.
- `src-crates/ffi` — the bindings. Dumb mirror.
- `build/` — packaging, devshell, `ws-*` gate scripts.

## Structure

- Capability = independently feature-flagged library module; `full`
  aggregates; product and bindings consume `full`.
- Capabilities get module tests; bindings get parity suites, never unit
  tests.
- Content-processing engine invocations each leave one metered audit
  record: duration, engine, output counts.
- Model weights are embedded or fetched into a configurable cache; nothing
  else is fetched at runtime.
- Storage is library-internal — never bound.
