# Communication

- Think like caveman. Talk like caveman. Don't waste token. (use caveman skill)
- If user input is posed as question, do not assume to implement.
- Focus on absolute simplicity / YAGNI. Don't make decisions or implementations before needed. Avoid technical debt.

# Project

- Greenfield. No legacy/regression padding. Refactor and break when useful.
- Repo-wide hard constraints live in project files.
- Architecture principles live in `.agents/PRINCIPLES.md`; binding.
- System structure lives in `.agents/ARCHITECTURE.md`.
- Code conventions live in `.agents/CODESTYLE.md`.
- Never hardcode app name in code/docs. Read from constants or toml for rebrand safety.

# API Style

- Actor types are agent nouns saying what they do, e.g. `TextEmbedder`, `TextReranker`, `LayoutDetector`, `OcrEngine`, `FileTypeDetector`.
- Options structs are `<Actor>Options`.
- Model enums stay domain-named, e.g. `EmbeddingModel`, `OcrDetectionModel`.
- Byte/path method pairs are `<verb>_bytes` and `<verb>_file` in Rust; Python FFI uses `<verb>_path` for path variants.
- Pipeline configuration is a flat plain-field options struct with defaults, not a builder.

# Workspace

- Monorepo uses nix.
- Current shell is devshell.
- Shell source: `build/shell-dev.nix`.
- Use workspace scripts for checking/building:
  - ./build/scripts/ws-all.sh # for all below scripts run in sequence
    - ./build/scripts/ws-check.sh # faster, only check for problems
    - ./build/scripts/ws-fix.sh # fast, auto-fix where possible with linters
    - ./build/scripts/ws-test.sh # slow and exhaustive, runs all tests

# Documentation

- Keep docstrings consistent in nature, be concise and descriptive of purpose/intent
- Ensure docstrings exist on all functions in file root, compiler cannot force but still need
