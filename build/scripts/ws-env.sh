#!/usr/bin/env bash
# Workspace environment and .env loading. [needs: bash; optional: sops]

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$script_dir/../.." && pwd)}"

# Export and enter the shared workspace root.
export PROJECT_ROOT
cd "$PROJECT_ROOT"

# Load encrypted secrets when both sops and the file are available.
if command -v sops >/dev/null 2>&1 && [ -f "$PROJECT_ROOT/.env.enc" ]; then
	set -a
	source <(sops -d "$PROJECT_ROOT/.env.enc")
	set +a
fi

# Load local plaintext environment overrides last.
if [ -f "$PROJECT_ROOT/.env" ]; then
	set -a
	source "$PROJECT_ROOT/.env"
	set +a
fi
