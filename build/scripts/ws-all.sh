#!/usr/bin/env bash
# Run fix, check, test, and parity scripts. [needs: cargo +rustfmt +clippy, cargo-deny, cargo-machete, uv]

set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ws-env.sh"

./build/scripts/ws-fix.sh
./build/scripts/ws-check.sh
./build/scripts/ws-test.sh
./build/scripts/ws-parity.sh
