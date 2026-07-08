# Step 03 — Chunking

Read `00-overview.md` first. No prior steps required (independent of 01/02).

## Goal

Create the internal chunker that `Index` uses to split record content into
chunks for vector retrieval, plus the public `ChunkingOptions` that
`IndexOptions` will embed in step 05.

Chunks are an internal retrieval mechanism. The chunker's output types stay
crate-private; only `ChunkingOptions` is public.

## Context

- No chunking exists in core today.
- Records arrive as plain text in this plan (extraction feeds richer input
  later).
- Consumers may disable chunking — then a record's full content is stored
  as a single vector entry.

## Design

New module `src-crates/core/src/index/` (created here; `Index` itself lands
in step 05):

- `mod.rs` — declares submodules; public surface for now is `ChunkingOptions`
- `chunking.rs` — implementation + crate-private types
- `chunking/tests.rs` or `#[cfg(test)] mod tests` per repo convention

```rust
/// Options controlling how record content is split for retrieval.
pub struct ChunkingOptions {
    /// Split content into chunks. When false, one chunk per record.
    pub enabled: bool,          // default true
    /// Target chunk size in characters.
    pub max_chars: usize,       // pick a sensible default (~1200–2000)
    /// Overlap between consecutive chunks in characters.
    pub overlap_chars: usize,   // default a modest fraction of max_chars
}
```

Behavior:

- Split on natural boundaries (paragraph, then sentence, then whitespace)
  before falling back to hard character cuts. Never split inside a UTF-8
  code point or (best effort) inside a word.
- Deterministic: same input + options → same chunks. Chunk identity is
  `(record_id, sequence)` — no randomness, no timestamps.
- Empty/whitespace content yields zero chunks (the record still exists via
  its title row and graph node).
- Pure functions, no async, no model or storage dependencies.

Keep the strategy simple; smarter strategies (token-aware, semantic) can
become new fields/enum variants later. Do not add them now.

## Scope

- `ChunkingOptions` with working defaults, the splitting implementation,
  exhaustive unit tests (boundaries, overlap, unicode, tiny/huge inputs,
  disabled mode).

## Out of scope

- Embedding, storage writes, `Index` wiring (step 05).
- Token-based or semantic chunking.

## Acceptance

- `./build/scripts/ws-check.sh` and `./build/scripts/ws-test.sh` pass.
- Docstrings on all public items.
- Only `ChunkingOptions` escapes the module.
