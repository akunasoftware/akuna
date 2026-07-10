# Principles

Hard rules. Binding on all work. Add sparingly, with owner sign-off.

- **No external runtimes.** Everything runs inside our own embedded stack —
  inference, storage, all of it. Exactly one embedded ML backend. Model
  weights in interchange formats are fine; external runtimes are not.
- **Simplest surface.** Every public item earns its place.
- **FFI is a dumb wrapper.** Export annotations and type conversion only;
  if a binding needs behavior, add a core API — never adapter code.
- **Bindings mirror core 1:1.** Same operations, names, and shapes in every
  language; a binding that must reshape core is a core-surface bug.
- **One shape per concept.** One type per piece of information, reused
  everywhere it appears; no partial copies. Serialization and schema
  support live on the core type.
- **Product stays out of core.** The knowledge service is implemented in
  app, on top of core — never in core.
- **Never hardcode the app name** in runtime or display strings; read it
  from the single app-name source. Package identifiers are exempt.
- **Configure only what genuinely warrants it.**
- **No premature optimization** — Knuth: "premature optimization is the
  root of all evil." Correct and simple first; optimize only what
  measurement proves hot.
- **Purpose, never mechanics.** Public docstrings say what a thing is for;
  internal items may state mechanics. Rationale lives in code comments,
  not docs.
