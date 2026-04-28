# Review Uncommitted

Review staged + unstaged changes.

## Ownership

- Subagents give feedback only.
- Main thread owns judgement, fixes, checks, reruns.
- If intent unclear, ask user.

## Subagents

- Use 2+ read-only review subagents.
- Run reviewers independently; do not share prior findings or expected failures.
- Input: `git status`, full staged diff, full unstaged diff, task intent if known.
- Output: `path:line: severity: issue. fix.` or `No findings.`
- Severities: `blocker`, `high`, `medium`, `low`.

## Review Lens

- Ignore `.agents/**` by default.
- Review `.agents/**` only when user explicitly asks, or when `.agents/**` is the only uncommitted context.
- Compare implementation against intended behavior, not only changed code.
- If intent is unclear from diff, code, tests, names, or docs: ask user before fixing.
- Find correctness bugs, regressions, missing edge cases, weak API shape, avoidable allocation/clone/copy, bad ownership, footguns.
- Apply loaded code style guidance.
- Ignore style-only nits unless meaning, safety, or maintainability changes.

## Fix Rules

- Fix high-value items only: correctness, compile break, test break, clear perf/ergonomic win.
- Keep smallest correct patch.
- Do not rewrite for taste.
- Do not revert unrelated user changes.

## Loop

- Inspect `git status` + full uncommitted diff.
- Run review subagents against current uncommitted diff.
- Triage subagent findings.
- Main thread fixes high-value items.
- Run required checks from loaded workflow guidance.
- Fix check failures you caused.
- If checks fail from pre-existing or unrelated user changes, report exact blocker and stop.
- Rerun review subagents against resulting diff.
- Repeat until no fixable findings remain and checks pass.
- Stop only when blocked, user clarification needed, or no fixes remain.

## Output

- Findings first, ordered by severity: `path:line: severity: issue. fix.`
- If fixed, include concise change summary.
- If no findings, say `No findings.` plus checks run.
