# Communication

- Think like caveman. Talk like caveman. Don't waste token. (use caveman skill)
- If user input is posed as question, do not assume to implement.

# Project

- Greenfield. No legacy/regression padding. Refactor and break when useful.
- Repo-wide hard constraints live in project files.
- Never hardcode app name in code/docs. Read from constants or toml for rebrand safety.

# Environment

- Monorepo uses nix.
- Current shell is devshell.
- Shell source: `build/shell-dev.nix`.

# Documentation

- Keep docstrings consistent in nature, be concise and descriptive of purpose/intent
- Ensure docstrings exist on all functions in file root, compiler cannot force but still need

# Brain

- Durable project memory lives in `.agents/brain/`.
- Main brain:
  - `.agents/brain/style.md`
  - `.agents/brain/codestyle.md`
  - `.agents/brain/workflow.md`
  - check for others, load as needed
- `opencode.json` autoloads main brain via `instructions`.
