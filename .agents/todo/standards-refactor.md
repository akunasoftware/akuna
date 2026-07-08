# Standards Refactor

Audit source: `.agents/PRINCIPLES.md`, `.agents/CODESTYLE.md`, and
`.agents/ARCHITECTURE.md`.

## Boundary And Surface

- [x] Replace public `anyhow` errors for detection, index, and layout with
  dedicated source-preserving errors.
- [x] Replace public `anyhow` errors for embedding and reranking.
- [x] Replace vendor Magika public types with core-owned detection shapes.
- [x] Move index metadata/filter types out of storage public APIs.
- [x] Make layout own byte/file decoding APIs.
- [x] Bind layout decoding and every FFI operation/name/shape 1:1.
- [x] Add missing model cache overrides to FFI layout and reranking options.
- [x] Move core relationship batch validation from app into `Index`.

## Models And Assets

- [x] Honor `cache_dir` for every OCR/layout weight and dictionary fetch.
- [x] Bundle and pin OCR/layout non-weight model assets.
- [x] Fetch embedding/reranking config, tokenizer, and weights into the
  configured Hugging Face cache.
- [x] Remove the layout environment weight override.
- [x] Share duplicated XLM-R position/padding invariants.
- [x] Propagate layout inference failures instead of fabricating tensors.

## Correctness And Determinism

- [x] Make record/chunk/graph updates atomic or roll back failed mutations.
- [x] Make vector FTS lifecycle and deletion transactional across reopen and
  option changes.
- [x] Add explicit stable tie-breaks for detection, reranking, layout, OCR,
  graph neighbors, and extraction pipeline outputs.
- [x] Use one shared hex-token helper.
- [x] Add required audit records/counts to extraction and index engines.
- [x] Add injectable test seams for index dependencies.

## Tests And Parity

- [x] Add complete parity proof for every native port and every exported FFI
  operation/output field.
- [x] Move FFI behavioral coverage into core; leave FFI suites for reference
  parity only.
- [x] Add missing OCR/layout module tests and move their inline tests to
  sibling test modules.
- [x] Apply sibling test-module shape to remaining modules.
- [x] Use the shared model-stack runner for all model-heavy tests.
- [x] Add standalone feature checks to workspace gates.

## Shape, Docs, And Tooling

- [x] Align remaining capability roots, imports, visibility, options
  validation, and dependency declarations with the code-style blueprint.
- [x] Add remaining runnable module doctests required by the convention.
- [x] Remove banned/stale tooling and broken OCI helpers; repair CI masking.
- [x] Update README capability/workspace tables.

## Audit Loop

- [x] Re-run independent standards audits after each fix batch until they
  report no remaining concrete feedback.
