# Cleanse

Execution command.
Not audit.
Edit now unless blocked by missing decision, external failure, or unsafe destructive action.

Goal: rip AI slop from current user scope.
Do not downscope to easy file.
Do not stop after first deletion, green check, or reviewer pass.
Loop until stop gate passes.

## Law

- Remove > add.
- Fewer files, knobs, names, concepts.
- Real product shape only.
- Fake architecture dies.
- Intentional abstraction lives when it simplifies caller API and hides implementation specifics.
- Greenfield default: break internal API if simpler.
- Compat must prove itself: persisted data, shipped CLI/API contract, or external consumer.
- Future implementation swaps are valid only behind one simple surface with config-selected sensible defaults.
- “Tests expect it”, “public”, “safer”, “maybe used” is not proof.

## Hunt

- Hidden features, flags, cfg soup.
- Hidden deps: runtime/model/native deps, lockfiles, build scripts, generated/downloaded artifacts, optional backends.
- Unused heavy deps after feature collapse.
- Backcompat padding.
- “Might need later” code.
- Duplicate APIs, duplicate extraction paths, duplicate outputs.
- Re-reading/re-parsing for parts + text.
- Text pipeline separate from structured parts.
- Fake metadata/ranges/provenance.
- Public deps/features that are implementation detail.
- `anyhow` leaking from lib public API.
- Untyped public errors where caller needs choice.
- Silent fallback changing requested behavior.
- Debug paths shipped as API.
- Broad `allow` / `expect` hiding debt.
- Hardcoded app/env/repo names.
- Generic names hiding domain.
- One-impl traits.
- One-impl traits pretending to be plug-in systems.
- Do not kill one-impl traits used as intentional implementation-hiding boundaries for simple public APIs.
- One-setting config.
- Wrapper types only renaming another type.
- Re-export chains hiding owner.
- Pattern-based module names, not domain names.
- Async/concurrency without parallel work.
- Folder layout where owner unclear.

## Test Slop

Tests not sacred.
Delete tests for removed features, deps, cfgs, fixtures.
Delete low-level probes, generated/model/runtime probes, artifact/snapshot/loader/backend trivia.
Delete tests protecting internal shapes, duplicate outputs, fake metadata, removed design.
If simpler design breaks slop test, change/delete test.
Replace only when real product behavior loses coverage.
Prefer one high-level behavior test over many micro/probe tests.

## Moves

- Delete first, repair compile.
- Collapse feature surface.
- Remove public feature flags for internals.
- Own feature/module at parent `mod`; no inner `cfg` soup.
- Remove code paths, flags, optional deps, env vars, docs, tests, fixtures, generated assets, stale names together.
- No disabled skeletons behind cfg/runtime option/trait/TODO.
- Move code to owner; do not add glue.
- Put generated/vendor/model artifacts under owner module.
- Split giant files only when ownership clearer.
- Never split by line count, technical layer, or style taste.
- Public API thin.
- Public API may expose domain trait/config while concrete backend stays hidden.
- Keep backend-specific types private unless caller must control backend-specific behavior.
- Orchestration separate from model/runtime internals.
- Structured parts source of truth.
- Derive text from parts; no parallel text pipeline.
- One canonical representation; derived views only.
- Remove duplicate `chunks`/`segments` concepts.
- Typed public lib errors.
- `anyhow` only deep loader/runtime internals.
- No silent fallback unless explicit product option.
- Inline single-use helpers.
- Collapse pass-through modules/re-export-only files.
- Collapse one-impl traits unless they are real boundaries hiding implementation details behind simple defaults.
- Remove fields caller can compute.
- Rename generic names toward domain.

## Subagents

Mandatory beyond one tiny file-local edit.
Use investigator before structural edits.
Use reviewer after edits/checks.
Skipping needs explicit user permission.
Do not use agents to delay edits.

Loop:

1. Investigator maps shape.
2. Reviewer attacks smells/regressions.
3. Builder does tiny bounded edits if useful.
4. Main thread refactors.
5. Check.
6. Reviewer again.
7. Repeat until stop gate passes.

Parallel agents when independent:

- Feature/cfg audit.
- API/error audit.
- Output shape audit.
- Folder ownership audit.
- Dead code/deps audit.
- Test slop audit.

Agents get narrow target.
Agents return files, smells, rip/consolidate moves.
Main thread decides and edits.

## Stop Gate

Minimum loop:

1. Map.
2. Edit.
3. Verify.
4. Reviewer.
5. Second edit or explicit no-op reason.
6. Verify again.

Stop only when all true:

- Workspace checks pass.
- Reviewer finds no actionable issue in current scope.
- Stale-name grep clean.
- No known actionable smell remains, or user accepts it.

## Verify

- Format.
- Focused feature checks.
- Sample commands if behavior changed.
- Grep removed names/features/cfgs/env vars/fields/concepts/user strings.
- Inspect `Cargo.toml`, `Cargo.lock`, build scripts, generated assets for removed deps/features.
- Run `./build/scripts/ws-all.sh`.
- Re-run reviewers after major rip.

Green check not stop permission.
If check fails, fix and rerun.
Focused checks do not replace workspace check.
Do not stop on failure unless missing decision or external dependency blocks.

## Final

Caveman terse.
Include:

- Ripped out.
- Consolidated.
- Breaking changes.
- Checks run.
- Remaining smell accepted by user.

No praise.
No long plan.
No “could also”.
No unexplored architecture.
