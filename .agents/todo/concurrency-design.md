# Concurrency Design

One backlog item: design concurrency for ML and indexing together.

Scope:

- ML inference scheduling.
- Batching and worker model.
- Index read/write behavior under concurrent API and FFI use.
- Index hydration/query batching under load.
- Storage engine concurrency expectations.

Constraints:

- Keep current ML calls sync and simple until this work happens.
- Do not add ad hoc locks, queues, or per-call worker pools meanwhile.
- Design this holistically, then implement once.
