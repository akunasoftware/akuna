# Step 03 — Chunking (Lift From Extraction)

Read `00-overview.md` first. No prior steps required (independent of 01/02).

## Goal

Core already has structure-aware segmentation — it lives inside extraction.
Lift it into its own `chunking` namespace, simplify it, and extend it into
the chunker `Index` will use. Do NOT write a parallel custom chunker: the
owner wants one segmentation capability, reused.

What exists today, all extraction-private:

- `extraction/extractors/code.rs` — tree-sitter leaf-node segmentation for
  ~30 languages (the structured-recognition path to preserve).
- `extraction/parts.rs::from_text` — the raw-text fallback: one part per
  non-empty line. Too naive to chunk with; this is what gets replaced.
- `extraction/types.rs` — `PartKind` (Heading, Paragraph, Code, Table,
  ListItem, …) and byte-range provenance.

## Design

New module `src-crates/core/src/chunking/` behind a new `chunking` cargo
feature (carries the tree-sitter deps; the `extraction` feature now depends
on it; step 05's `index` feature will too). Layout per repo convention
(`mod.rs`, submodules, `#[cfg(test)] mod tests;` + sibling `tests.rs` as in
`embedding/`).

**Segmentation (moved + improved).** The unit is a segment: text, a kind,
and its byte range in the source.

- Move the tree-sitter machinery from `extractors/code.rs` here
  (language-by-extension table, leaf-named-node ranges) as the code
  strategy.
- Replace the line-based fallback with real prose segmentation: split on
  blank-line paragraph boundaries — whole paragraphs, however large.
  Sentence-level splitting is the PACKER's job (intra-segment split); the
  segmenter has no size threshold and never sees `ChunkingOptions`.
- API shape (pinned — mirrors the existing dispatch in
  `document.rs::content_from_text` so extraction's engine labels and
  `>1 parts` gate survive): two entries, not one router. The code strategy
  returns `Option<Vec<Segment>>` (None = unsupported extension / parse
  error), the prose strategy is infallible. Extraction keeps its
  try-code-then-fallback dispatch and its `"tree-sitter"` pipeline label.
  Segments carry raw (untrimmed) source text + kind + byte range; the
  segmenter drops whitespace-only segments; callers do their own display
  trimming.
- `Index` has plain text and no hint — prose strategy. Where no hint
  exists but the content is clearly code, the existing detection
  capability MAY route to the code strategy — but `FileTypeDetector` is an
  async model-loading actor and the segmenter is pure sync functions, so
  this does NOT slot in cleanly at this layer: leave it as a documented
  follow-up for `Index` (which is already async and owns models) to do
  detection-assisted routing itself later. Do not wire ML into `chunking`.
- Kind enum: `PartKind` moves to `chunking` KEEPING its name; extraction
  re-exports it (`pub use crate::chunking::PartKind;`) so
  `akuna_core::extraction::PartKind`, the FFI conversions, and app JSON
  keep working unchanged. Serde derives travel with it (serde is a
  non-optional core dep — no gating issue).
- Prose fallback parts change shape deliberately: paragraph segments emit
  `PartKind::Paragraph` with byte-range provenance (the old line splitter
  emitted `Text` with no provenance). App JSON output changes accordingly
  — sanctioned, part of the intended improvement.

**Extraction consumes it.** `extraction/parts.rs` and `extractors/code.rs`
become thin calls into `chunking`; extraction keeps owning
`ExtractionPart`/provenance assembly (part index, page, bbox). Extraction
behavior may improve (paragraph parts instead of line parts from the
fallback) — update extraction tests accordingly; that's an intended
simplification, not a regression.

**Chunk packing (new, for `Index`).** On top of segments, a packer that
produces retrieval-sized chunks:

```rust
/// Options controlling how record content is split for retrieval.
pub struct ChunkingOptions {
    pub enabled: bool,        // default true; false = one chunk per record
    pub max_chars: usize,     // default 1600 — hard upper bound per chunk
    pub overlap_chars: usize, // default 200 — used only when splitting inside a segment
}
```

Pinned semantics (these resolve real implementer forks — keep them):

- "Characters" everywhere = Unicode scalar values (`char` count), never
  bytes. Cuts happen on `char` boundaries; word-boundary snapping is best
  effort (CJK and unbroken runs hard-cut at `max_chars`).
- Packing: greedily fill each chunk with whole segments up to `max_chars`.
  A segment that alone exceeds `max_chars` is split at sentence, then
  whitespace, then hard cuts; only these intra-segment splits get
  `overlap_chars` of tail overlap (segment boundaries are already
  semantic — no overlap across them). Overlap counts toward the receiving
  chunk's budget, so `max_chars` stays a hard bound.
- Deterministic: same input + options → same chunks. Chunk identity is
  positional — `Index` (step 05) forms `(record_id, sequence)` from the
  output order. The chunker never sees record ids.
- Empty/whitespace content → zero chunks. `enabled: false` → one chunk
  containing the full trimmed content.
- Degenerate options are clamped (effective `max_chars ≥ 1`, effective
  `overlap < max_chars`) rather than errored — pure infallible functions;
  rationale as a `//` comment.
- Manual `Default` impl (derived defaults would be zero/false).

**Public surface:** `ChunkingOptions` and the kind enum are `pub`; the
segment entries extraction consumes and the ENTIRE packer are `pub(crate)`
(a `pub fn` cannot return crate-private types, and chunks never cross the
public API). Docstrings purpose-only; mechanics like the boundary
hierarchy live in `//` comments. Prefer plain string-literal packer tests
over corpus fixtures — avoids widening the `testkit` feature gate.

## Scope

- The `chunking` module + feature, the move out of extraction, extraction
  rewired and its tests updated, the prose strategy, the packer, and
  exhaustive packer tests: boundary hierarchy, overlap only within split
  segments, unicode/emoji cut safety, tiny/huge inputs, unbroken runs,
  disabled mode, empty input, determinism, clamping.
- `Cargo.toml`: `chunking` feature owning the tree-sitter deps;
  `extraction` depends on `chunking`; both in `full` so `--all-features`
  CI covers them.

## Out of scope

- Embedding, storage writes, `Index` wiring (step 05).
- Token-aware or semantic (ML) chunking.
- New segmentation strategies beyond code + prose (+ optional
  detection-assisted routing).

## Acceptance

- `./build/scripts/ws-check.sh` and `./build/scripts/ws-test.sh` pass,
  including updated extraction tests.
- Standalone feature builds pass (CI only runs `--all-features`, which
  hides gating bugs): `cargo check -p akuna-core --no-default-features
  --features chunking` and `--features extraction`.
- `extraction` compiles with no segmentation logic of its own — it calls
  `chunking`.
- Only the intended types escape the module; no chunk-shaped output is
  public.
