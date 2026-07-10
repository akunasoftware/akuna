# Prompt Refresh

Review prompt docs:

- `AGENTS.md`
- `.agents/PRINCIPLES.md`, `.agents/CODESTYLE.md`, `.agents/ARCHITECTURE.md`
- `.agents/opencode.json`
- `.agents/commands/**/*.md`

## Process

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
- Keep `AGENTS.md` lean: entry point + doc index only.
  - Doc files must be listed literally in `AGENTS.md` for discovery.
  - Rule files must be autoloaded through `opencode.json` `instructions`.
  - Do not rely on `AGENTS.md` `@file` refs; they do not auto-parse.
- One owner per rule; subprompts must not repeat top-prompt rules.
- Forbid technology brands and literal code/source refs inside prompt docs.
  - Code paths move and stacks change; stale refs mislead agents.
  - Prefer roles and discovery, e.g. "the embedded graph engine", "find
    current examples". Verbatim commands and doc addresses are exempt.
- If a prompt grows large, distill into a new prompt or namespace.
- Durable domain intent goes to the owning doc file, one topic per place.
