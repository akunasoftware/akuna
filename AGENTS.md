# Communication

- Think caveman. Talk caveman. Don't waste token.
- Question ≠ implement request.
- YAGNI. Nothing before needed.

# Project

- Greenfield. No legacy padding. Break freely.

# Docs

- Docs are authoritative.
- `.agents/PRINCIPLES.md` — hard rules.
- `.agents/CODESTYLE.md` — code conventions.
- `.agents/ARCHITECTURE.md` — demanded shape.

# Workspace

- Devshell: `nix develop`.
- Done = `./build/scripts/ws-check.sh` + `ws-test.sh`; FFI also
  `ws-parity.sh`. `ws-fix.sh` auto-fixes; `ws-all.sh` = everything.
- No watch-mode tools — they hang the CLI. No `lsp-ignore` without consent.
- Fix all errors before stopping.
