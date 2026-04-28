# Full Review

Run a repo-wide review using read-only subagents, then verify findings in main thread.

Use this when user asks for full review, stale artifact hunt, docs/API consistency review, or broad cleanup pass.

## Ownership

- Subagents review only.
- Main thread verifies findings, decides relevance, performs edits, runs checks.
- Do not let subagents run build/test/check commands.
- If user only asked for review, do not edit unless they explicitly ask.

## Scope Setup

- Inspect current repo shape first:
  - root README and crate READMEs
  - Cargo workspace and feature flags
  - public source modules
  - build/config scripts
- Capture explicit user exclusions before spawning reviewers.
- Repeat exclusions in every subagent prompt.

Common exclusions learned from this repo:

- Ignore `.agents` package/node artifacts unless user explicitly asks.
- Ignore OCI `serve` command when user says server will be added later.
- Treat graph `Assertion` / `Provenance` WIP shape as intentionally allowed unless docs imply stability.
- Do not flag generated/build outputs under `target/` or `.git/`.

## Subagents

Spawn 3 blind reviewer subagents in parallel:

- Docs reviewer:
  - `README.md`, `src-crates/**/README.md`, licenses, prompt docs.
  - Check stale claims, broken links, bad markdown rendering, examples that do not match public API.
- Source/API reviewer:
  - `src-crates/**/*.rs`, crate `Cargo.toml` files.
  - Check feature-gate bugs, public API/docs mismatch, broken examples, dead modules, benches, hardcoded local paths.
- Config/build reviewer:
  - root `Cargo.toml`, `flake.nix`, `build/**`, `bacon.toml`, `deny.toml`, `.gitignore`, `opencode.json`.
  - Check stale paths, package metadata drift, build dependency drift, script name mismatch.

Require output format:

```text
path:line: severity: problem. fix.
```

Or:

```text
No findings.
```

## Prompt Rules

- State exact focus paths.
- State exact exclusions.
- Say “Do not edit. Do not run checks.”
- Say “Return findings only.”
- Mention known project intent when relevant, so agents do not flag intentional WIP.
- Avoid telling agents expected findings; they should review blind.

## Main Thread Verification

- Re-read every reported file/line before accepting finding.
- Check if finding is true, stale, already excluded, or reviewer overreach.
- Verify snippets against actual public API when reviewers flag docs.
- Verify feature-gate findings against `lib.rs`, `types/mod.rs`, and Cargo features.
- Do not trust severity blindly; re-rank by actual impact.

## Known Agent Behavior

- Reviewers are good at finding stale docs, broken examples, feature-gate bugs, hardcoded paths, and config drift.
- Reviewers may over-flag known future work unless exclusions are repeated clearly.
- Reviewers may flag `.agents` package artifacts; exclude them unless task targets agent tooling.
- Reviewers may miss whether code examples are intended as continuations; ask for self-contained examples when docs are README-like.
- Reviewer severity can be too high for docs wording and too low for feature-gate compile breaks; main thread must re-rank.
- Reviewers can find real source bugs outside docs cleanup scope; report them separately if user did not ask for fixes.

## Optional Fix Loop

If user asks to fix findings:

- Fix smallest correct patch.
- Keep unrelated user changes intact.
- Run workspace check from workflow memory: `./build/scripts/ws-all.sh`.
- Fix failures caused by your changes.
- Rerun focused reviewers only on changed/fixed areas.

## Final Output

- If review only: list verified findings, grouped by severity.
- If fixes performed: list changes and checks run.
- Include ignored/excluded areas so user knows omissions were intentional.
- Keep residual risks explicit.
