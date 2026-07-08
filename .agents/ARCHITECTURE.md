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
  tests. Python parity suites compare custom implementations against named
  upstream Python references; core owns behavioral tests.
- Content-processing engine invocations each leave one metered audit
  record: duration, engine, output counts.
- Model configs, tokenizers, and weights are fetched into a configurable
  cache; inference runs entirely in the embedded stack.
- Storage is library-internal — never bound.
