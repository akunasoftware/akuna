#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$script_dir/../.." && pwd)}"

export PROJECT_ROOT

cd "$PROJECT_ROOT"

# if present, load sops-encrypted secrets into session env
if command -v sops >/dev/null 2>&1 && [ -f "$PROJECT_ROOT/.env.enc" ]; then
  set -a
  source <(sops -d "$PROJECT_ROOT/.env.enc")
  set +a
fi

# if present, load unencrypted .env into session env
if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi
