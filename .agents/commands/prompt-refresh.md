# Prompt Refresh

Review prompt docs:

- `AGENTS.md`
- `opencode.json`
- `.agents/commands/**/*.md`
- `.agents/skills/**/*.md` if present
- `.agents/brain/**/*.md` if present

## Process

- Apply communication + docs guidance from loaded brain.
- Use 2+ blind read-only subagent reviewers.
  - Give realistic task only; do not mention expected guide path.
  - Run reviewers independently; do not share prior findings or expected failures.
  - Require `file:line: severity: issue. fix.` or `No findings.`
  - Failure = missed guide, wrong guide use, lost fidelity, unclear workflow, or bad markdown.
- If subagent misses guide or misuses guide: fix prompt.
- Validate with fresh subagent after fix.
- Loop until no fixable prompt failure remains.

## Rules

- Markdown good for humans + agents.
  - Keep related commentary indented.
- Keep `AGENTS.md` lean.
  - Root = project constraints + command/brain index only.
  - Main brain items must be listed literally in `AGENTS.md` for human discovery.
  - Main brain items must be autoloaded through `opencode.json` `instructions`.
  - Do not rely on `AGENTS.md` `@file` refs; OpenCode docs say AGENTS does not auto-parse them.
- Avoid duplication.
  - Subprompts must not repeat top-prompt rules unless needed for isolation.
- Keep docs human-readable.
- Forbid literal code/source file refs inside markdown prompts/brain.
  - Code paths move; stale refs mislead agents.
  - Prefer discovery instructions, e.g. “find current examples”.
  - Brain refs are OK.
  - Keep core brain index in `AGENTS.md` and `opencode.json` `instructions`.
- If prompt grows large, distill into new prompt or namespace.
- Move durable project/domain intent to `.agents/brain/`.
  - Use one topic per file.
  - Name by domain noun: `billing.md`, `branding.md`, `deployment.md`.
  - Keep structure clean and growing.
