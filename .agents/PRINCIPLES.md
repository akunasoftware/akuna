# Principles

Hard rules. Owner sign-off to change.

- **No external runtimes.** Everything runs inside our own embedded stack —
  inference, storage, all. Exactly one embedded ML backend. Weights in
  interchange formats fine; external runtimes not.
- **Simplest surface.** Every public item earns its place.
- **FFI is a dumb wrapper.** Annotations + type conversion only. Binding
  needs behavior → add core API.
- **Bindings mirror core 1:1.** Same operations, names, shapes everywhere.
  Reshaping = core-surface bug.
- **One shape per concept.** No partial copies. Serialization/schema
  support lives on the core type.
- **Product stays out of core.**
- **Never hardcode the app name** in runtime or display strings; one
  source. Package identifiers exempt.
- **Configure only what genuinely warrants it.**
- **No premature optimization** — Knuth: "premature optimization is the
  root of all evil." Optimize only what measurement proves hot.
- **Purpose, never mechanics.** Public docstrings say what for; internals
  may say how. Rationale in code comments, not docs.
