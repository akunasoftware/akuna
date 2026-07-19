# Concurrency Design

One backlog item: design concurrency for ML.

Scope:

- ML inference scheduling.
- Batching and worker model.

Constraints:

- Keep current ML calls sync and simple until this work happens.
- Do not add ad hoc locks, queues, or per-call worker pools meanwhile.
- Design this holistically, then implement once.
