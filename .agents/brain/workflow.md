# Workflow

- After code changes, make repo build, lint, test without warnings.
- Workspace check command: `./build/scripts/ws-all.sh`.
- Check `bacon.toml` `all` job for workspace checks.
- Do not run `bacon`; watcher hangs CLI.
- Never `lsp-ignore` without explicit developer consent.
- Fix all errors before stopping, unless feedback needed.
