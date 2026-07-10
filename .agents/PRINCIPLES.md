# Principles

Hard rules. Binding on all work. Add sparingly, with owner sign-off.

## Purpose

The whole knowledge-tooling toolset in one embedded package: sophisticated
functionality behind the simplest possible API surface, distributable
anywhere, usable from any language. It exists because nothing else does the
whole job in one embedded, runtime-free artifact.

## Rules

- **No external runtimes.** Nothing may depend on ONNX Runtime, PyTorch, or
  any runtime outside our own embedded one — not for inference, not for
  storage, not for anything. Burn is the sole ML backend. (Model weights in
  ONNX *format* are fine; the runtime is not.)
- **Simplest surface.** Expose the minimum API that serves the purpose;
  every public item earns its place.
- **FFI is a dumb wrapper.** Export annotations and type conversion only.
  If a binding needs behavior, add a core API — never adapter code.
- **Bindings mirror core 1:1.** Same operations, names, and shapes in every
  language. A binding that must reshape core is a core-surface bug.
- **One shape per concept.** One type per piece of information, reused
  everywhere it appears; no partial copies. Serialization and schema
  derives live on the core type for this reason.
- **Product stays out of core.** The knowledge service is implemented in
  app, on top of core — never in core.
- **Configure only what genuinely warrants it** — configurability is
  complexity.
- **No premature optimization** — Knuth: "premature optimization is the
  root of all evil." Correct and simple first; optimize only what
  measurement proves hot, when it actually matters.
- **Purpose, never mechanics.** Public docstrings say what a thing is for —
  no implementation details or behavior specifics; those rot. Internal
  items may state mechanics. Rationale that must survive lives in `//`
  code comments, not docs.
