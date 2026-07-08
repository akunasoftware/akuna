# Step 08 — Result Previews

Read `00-overview.md` first. Requires step 06 (07 not required).

## Goal

Fill `IndexSearchResult.preview`: a manually built, semantically relevant
truncated string centered on the text that made the record match. Users
scanning results should see *why* a record hit without reading content.

## Context

- `IndexSearchResult.preview: Option<String>` exists since step 06, always
  `None`.
- The pipeline already knows each record's best-matching evidence (its
  top-scoring chunk, or its title for title-only matches).
- Chunk text is stored in the vector layer; chunks stay hidden — a preview
  is a plain string, never a chunk object.

## Design

Preview construction (pure string logic, no model calls):

1. Take the record's best-matching chunk text (post-rerank when enabled,
   post-fusion otherwise).
2. Locate the most query-relevant region inside it — term-overlap scoring
   over a sliding window is sufficient; no new ML.
3. Cut a window centered on that region: snap to word boundaries, prepend/
   append ellipses when truncated mid-content, collapse internal whitespace.
   Never split UTF-8 code points.
4. Title-only matches (or records whose evidence is unavailable) fall back
   to the leading content window.

Configuration: previews are on by default with a fixed sensible length
(~200–300 chars). If a knob is genuinely needed, one typed field on
`IndexSearchQuery` (e.g. `preview: bool` or an options enum) — keep it
minimal, no formatting options.

Records pulled in by graph expansion (07) without retrieval evidence use
the fallback path.

FFI: surface is unchanged unless a query field is added; extend
`test_parity_index.py` to assert previews arrive in Python and center on
relevant text.

## Scope

- Preview builder + wiring into result assembly, unit tests on the string
  logic (unicode, short content, relevance centering, fallbacks), pipeline
  tests asserting populated previews, parity additions.

## Out of scope

- Highlight markers, multi-span previews, model-generated summaries.

## Acceptance

- `./build/scripts/ws-check.sh`, `ws-test.sh`, `ws-parity.sh` pass.
- Searching a long record returns a preview containing the query-relevant
  passage, not just the head of the content.
- Preview never exceeds the configured length bound.
