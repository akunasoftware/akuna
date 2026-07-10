# Communication

- Think like caveman. Talk like caveman. Don't waste token.
- If user input is posed as question, do not assume to implement.
- Absolute simplicity / YAGNI. No decisions or implementations before needed.

# Project

- Greenfield. No legacy/regression padding. Refactor and break when useful.
- Where existing code deviates from these docs, the docs win — align on next
  touch.
- Never hardcode the app name in runtime or display strings; read it from a
  constant or toml. Package/crate identifiers and registry names are exempt.

# Docs

- `.agents/PRINCIPLES.md` — binding hard rules. Read before designing.
- `.agents/CODESTYLE.md` — code conventions. Read before writing code.
- `.agents/ARCHITECTURE.md` — map of what exists.
- `.agents/planning/` — implementation specs. `.agents/commands/` — procedures.

# Workspace

- Nix monorepo: work inside the devshell (`nix develop`; source
  `build/shell-dev.nix`).
- Done = `./build/scripts/ws-check.sh` + `ws-test.sh` pass; FFI changes also
  `ws-parity.sh` (CI runs all three). `ws-fix.sh` auto-fixes; `ws-all.sh`
  runs everything.
- Do not run `bacon` — the watcher hangs the CLI. Never `lsp-ignore` without
  explicit consent.
- Fix all errors before stopping, unless feedback is needed.
