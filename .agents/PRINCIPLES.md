# Principles

Hard rules. Owner sign-off to change.

- **No external runtimes.** Everything runs inside our own embedded stack —
  inference, storage, all. Exactly one embedded ML backend. Weights in
  interchange formats fine; external runtimes not.
- **Simplest surface.** Every public item earns its place.
- **Bindings are dumb 1:1 mirrors.** Annotations + type conversion only;
  same operations, names, shapes everywhere. Binding needs behavior → add
  core API; reshaping = core-surface bug.
- **One shape per concept.** No partial copies. Serialization/schema
  support lives on the core type.
- **Product stays out of core.**
- **Never hardcode the app name** in runtime or display strings; one
  source. Package identifiers exempt.
- **Configure only what genuinely warrants it.**
- **No premature optimization** — Knuth: "premature optimization is the
  root of all evil." Optimize only what measurement proves hot.
- **Parity-proven ports.** Equivalence proven against upstream — committed
  goldens, or live reference with committed measured tolerance floors —
  never eyeballed.
- **One helper per invariant.** Subtle or risky operations live in one
  shared helper; call it, never reimplement.
- **Convert at boundaries.** Foreign types, errors, and values never cross
  a boundary raw; one conversion site per boundary.
- **Config reaches every site.** Every caller-set option is honored
  everywhere it applies; no silent defaults mid-stack.
- **Seams built in.** Dependencies injectable; tests deterministic.
- **One blueprint per module kind.** Modules of a kind share one shape;
  new ones copy it.
