# Step 08 — Result Previews

Read `00-overview.md` first. Requires steps 06 and 07 (steps run in
order, and the expanded-record preview test below needs 07's expansion).

## Goal

Fill `IndexSearchResult.preview`: a manually built, semantically relevant
truncated string centered on the text that made the record match. Users
scanning results should see *why* a record hit without reading content.

## Context

- `IndexSearchResult.preview: Option<String>` exists since step 06, always
  `None`.
- The pipeline's candidate type carries each record's KIND-TAGGED best
  evidence (pinned in 06): `Chunk(text)` from the FIRST rerank (or fused
  rank when reranking is off), `Title` for title-only hits, or
  `LeadingWindow(text)` for step 07 expanded records. Previews branch on
  the kind and use that first-rerank/fusion evidence — NOT step 07's
  representative-text scores.
- Result assembly already batch-hydrates from the vector layer
  (`get_records`); leading content comes from the record row's `content`
  there. No graph reads.
- Chunks stay hidden — a preview is a plain string, never a chunk object.

## Design

Preview construction (pure string logic, no model calls, no new deps),
applied only to the final ≤`limit` results:

1. Evidence text = the carried `Chunk`/`LeadingWindow` text; for `Title`
   or no evidence, use the head of the record's content instead (retained
   by 06's hydration).
2. Locate the most query-relevant region: lowercase BOTH query and
   evidence, split on whitespace/punctuation, dedup query terms (pin this
   simple tokenization — no stemming, no stopwords; a term is "present"
   by token equality, not substring). Slide the window across the
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
ellipses: budget the sliding window at 240 minus the chars any ellipses
will add, so the final string never exceeds 240. No configuration surface,
no query field ("configure only what genuinely warrants it"); FFI types
are unchanged.

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
