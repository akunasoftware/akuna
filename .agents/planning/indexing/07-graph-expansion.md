# Step 07 — Graph Expansion

Read `00-overview.md` first. Requires step 06.

## Goal

Fill the expansion seam in the search pipeline: traverse relationships from
candidate records to pull in related records, then produce the final
aggregate ranking. This is what makes results a holistic packet rather than
isolated hits.

## Context

- Search pipeline (06) with a passthrough expansion stage between record
  roll-up and final ordering.
- `GraphDbContext::neighbors` (04) returns connected nodes with their edges.
- Relationships are caller-supplied today (e.g. parent/cites); extraction
  will soon add file-path-derived links, NER entities later — expansion
  must not assume any particular predicate vocabulary.

## Design

Expansion stage (runs only when `graph: true`):

1. Take the top candidate records after roll-up (a bounded subset, e.g.
   top `limit`-ish — implementer picks and documents).
2. Fetch neighbors via the graph layer. One hop by default; if a depth knob
   is wanted, it goes on `IndexOptions` as a typed field with default 1 —
   do not add it speculatively if one hop suffices for v1.
3. Respect the query's collection scope and metadata filter: expanded
   records outside the searched collections or failing the filter are
   dropped (the metadata rule from `00-overview.md` — the graph stores the
   same metadata precisely so it can enforce this).
4. Merge expanded records into the candidate set, deduplicating against
   existing candidates.

Final aggregation — DESIGN DECISION for the implementer, with owner's
leaning recorded:

- **Option A (owner leans this way): rerank on record content.** Score every
  post-expansion candidate (retrieved and expanded alike) by reranking the
  query against record content. Content can be long — rerank against a
  bounded representative text (title + leading content, or title + the
  record's best-matching chunk text when it has one) rather than unbounded
  full content. Uniform treatment; expanded records get a real relevance
  score.
- **Option B: keep roll-up scores, derive expanded scores.** Retrieved
  records keep their step-06 scores; expanded records inherit a damped
  fraction of the score of the record that pulled them in. Cheaper (no
  second rerank pass) but scores are less principled.

Pick one, implement it, and record the rationale as a code comment at the
decision site. If reranking is disabled in options, Option B's derivation
is the fallback either way.

Results and identity: expanded records become ordinary
`IndexSearchResult`s. No flag distinguishing them — a result is a result.

## Scope

- Expansion stage, final aggregation, dedup, filter enforcement, config
  wiring, core tests (linked records surface; filters prune expansion;
  `graph: false` skips the stage; cycles/self-links terminate).
- FFI parity additions only if the public surface changed (e.g. a depth
  option); otherwise extend Python search parity with a relationship case.

## Out of scope

- NER, entity nodes, multi-hop ranking sophistication, edge weighting.

## Acceptance

- `./build/scripts/ws-check.sh`, `ws-test.sh` pass (`ws-parity.sh` if FFI
  surface changed).
- A search whose best hit links to a related record returns both, ranked
  sensibly.
- Expansion is inert when records have no relationships (no result change,
  no measurable cost blow-up).
