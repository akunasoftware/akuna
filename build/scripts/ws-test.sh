#!/usr/bin/env bash
# cargo test workspace. [needs: cargo]

set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ws-env.sh"

cargo test --workspace --all-features
