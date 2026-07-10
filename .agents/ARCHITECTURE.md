# Architecture

Demanded shape. Code aligns to this.

## Purpose

The whole knowledge-tooling toolset in one embedded package: sophisticated
functionality, simplest possible API, distributable anywhere, usable from
any language.

## Split

- `src-crates/core` — the library. All capabilities, all intelligence.
- `src-crates/app` — the product. CLI + HTTP API on the library; product
  concerns never enter the library.
- `src-crates/ffi` — the bindings. Dumb mirror, proven by parity suites
  per bound language.
- `build/` — packaging, devshell, `ws-*` gate scripts.

## Structure

- Capability = independently feature-flagged library module; `full`
  aggregates; product and bindings consume `full`.
- Every capability has module tests; the bindings surface is proven by
  parity suites, not unit tests.
- Model weights are embedded or fetched into a configurable cache; nothing
  else is fetched at runtime.
- Storage is library-internal — never bound.
