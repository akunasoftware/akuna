# Architecture

The demanded shape of the system. Code aligns to this.

## Purpose

The whole knowledge-tooling toolset in one embedded package — sophisticated
functionality behind the simplest possible API surface, distributable
anywhere, usable from any language.

## The split

- `src-crates/core` — the library. Every capability lives here, embedded;
  all intelligence belongs to the library and nothing else.
- `src-crates/app` — the product. CLI + HTTP API built on the library;
  product concerns never enter the library.
- `src-crates/ffi` — the bindings. A dumb mirror of the library, proven by
  parity test suites in every bound language.
- `build/` — packaging, the devshell, and the `ws-*` scripts that gate all
  work.

## Structure demands

- Each capability is an independently feature-flagged library module,
  aggregated in `full`; the product and bindings consume `full`.
- Every capability carries its own module tests; the bindings surface is
  proven by parity suites, not unit tests.
- Model capabilities embed their weights or fetch them into a configurable
  cache; nothing else is fetched at runtime.
- Storage is library-internal — never exposed through the bindings.
