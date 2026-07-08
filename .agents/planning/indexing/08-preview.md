# Step 08 — Result Previews

Read `00-overview.md` first. Requires step 06 (07 not required, but its
evidence contract is honored if present).

## Goal

Fill `IndexSearchResult.preview`: a manually built, semantically relevant
truncated string centered on the text that made the record match. Users
scanning results should see *why* a record hit without reading content.

## Context

- `IndexSearchResult.preview: Option<String>` exists since step 06, always
  `None`.
- The pipeline's candidate type carries each record's best-evidence text
  (pinned in 06): its top-scoring chunk text from the FIRST rerank (or
  fused rank when reranking is off), its title for title-only hits, or the
  leading-content window for step 07 expanded records. Previews use that
  first-rerank/fusion evidence — NOT step 07's representative-text scores.
- Result assembly already batch-hydrates from the vector layer
  (`get_records`); leading content comes from the record row's `content`
  there. No graph reads.
- Chunks stay hidden — a preview is a plain string, never a chunk object.

## Design

Preview construction (pure string logic, no model calls, no new deps),
applied only to the final ≤`limit` results:

1. Evidence text = the record's carried best-evidence text; when it's a
   title or missing, use the head of the record's content instead.
2. Locate the most query-relevant region: lowercase the query, split on
   whitespace/punctuation, dedup terms (pin this simple tokenization — no
   stemming, no stopwords). Slide a preview-width window across the
   evidence stepping by words; score = count of distinct query terms
   present; ties → total occurrences, then earliest window.
3. Whitespace-collapse the evidence FIRST (newlines/runs → single spaces),
   then select and cut — length math and centering operate on the
   collapsed text. Snap cut edges to word boundaries best-effort; prepend/
   append `…` when the cut doesn't reach the respective end. All indexing
   on `char` boundaries.
4. Zero query-term overlap → head of the evidence text (it is still the
   record's best-matching text).
5. Empty content and no evidence → `preview` stays `None`. That is the
   ONLY case allowed to stay `None` after this step.

Length: fixed `const PREVIEW_MAX_CHARS: usize = 240` — measured in Unicode
scalar values (matches Python `len()` for parity assertions), INCLUSIVE of
ellipses. No configuration surface, no query field ("configure only what
genuinely warrants it"); FFI types are unchanged.

## Scope

- Preview builder + wiring into result assembly; unit tests on the string
  logic (unicode/emoji cut safety, evidence shorter than the window,
  centering on a late passage, whitespace collapse, length bound inclusive
  of ellipses, zero-overlap fallback, title-only fallback); pipeline tests
  asserting populated previews including for an expanded record; extend
  `test_parity_index.py` to assert previews arrive in Python, respect the
  length bound, and contain the query-relevant passage.

## Out of scope

- Highlight markers, multi-span previews, model-generated summaries,
  preview options.

## Acceptance

- `./build/scripts/ws-check.sh`, `ws-test.sh`, `ws-parity.sh` pass.
- Searching a long record returns a preview containing the query-relevant
  passage, not just the head of the content.
- `len(preview) <= 240` (chars) always; `preview` is `Some` for every
  result whose record has content.
