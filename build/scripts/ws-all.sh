#!/usr/bin/env bash
# Run fix, check, test, and parity scripts. [needs: cargo +rustfmt +clippy, cargo-deny, cargo-machete, jq, uv]

set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ws-env.sh"

# Apply cheap formatting before compiler-backed fixes.
./build/scripts/ws-fix.sh

# Run static checks before executing tests.
./build/scripts/ws-check.sh

# Run Rust tests before the heavier cross-language suite.
./build/scripts/ws-test.sh

# Build bindings and run Python parity tests last.
./build/scripts/ws-parity.sh
