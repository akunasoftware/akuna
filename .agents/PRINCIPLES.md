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
- **Parity-proven ports.** Ported or mirrored functionality is proven
  equivalent — committed goldens, measured tolerance floors — never
  eyeballed.
- **One helper per invariant.** Subtle or risky operations live in one
  shared helper; call it, never reimplement.
- **Convert at boundaries.** Foreign types, errors, and values never cross
  a boundary raw; one conversion site per boundary.
- **Config reaches every site.** Every caller-set option is honored
  everywhere it applies; no silent defaults mid-stack.
- **Seams built in.** Every unit takes its dependencies injectable; tests
  are hermetic and deterministic.
- **Record engine work.** Every engine invocation leaves one metered audit
  record.
- **Capability boundaries compiler-enforced.** Visibility scoped to the
  capability; crate-wide only when genuinely shared.
- **One blueprint per module kind.** Modules of a kind share one shape;
  new ones copy it.
