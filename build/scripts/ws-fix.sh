#!/usr/bin/env bash
# Auto-fix workspace format and lints. [needs: cargo +rustfmt +clippy]

set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ws-env.sh"

# Format source without compiling.
cargo fmt --all

# Apply compiler suggestions across the workspace.
cargo fix --quiet --workspace --all-features --all-targets --allow-dirty --allow-staged

# Apply the more expensive Clippy suggestions last.
cargo clippy --quiet --fix --workspace --all-features --all-targets --allow-dirty --allow-staged
