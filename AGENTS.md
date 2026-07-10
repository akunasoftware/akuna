# Communication

- Think like caveman. Talk like caveman. Don't waste token.
- If user input is posed as question, do not assume to implement.
- Absolute simplicity / YAGNI. No decisions or implementations before needed.

# Project

- Greenfield. No legacy/regression padding. Refactor and break when useful.
- Where code deviates from these docs, the docs win — align on next touch.

# Docs

- `.agents/PRINCIPLES.md` — binding hard rules.
- `.agents/CODESTYLE.md` — code conventions.
- `.agents/ARCHITECTURE.md` — map of what exists.
- `.agents/planning/` — implementation specs. `.agents/commands/` — procedures.

# Workspace

- Work inside the devshell: `nix develop` (defined under `build/`).
- Done = `./build/scripts/ws-check.sh` + `ws-test.sh` pass; FFI changes also
  `ws-parity.sh`. `ws-fix.sh` auto-fixes; `ws-all.sh` runs everything.
- No watch-mode tools — watchers hang the CLI. Never
  `lsp-ignore` without explicit consent.
- Fix all errors before stopping, unless feedback is needed.
