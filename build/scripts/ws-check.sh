#!/usr/bin/env bash
# Static workspace checks. [needs: cargo +rustfmt +clippy, cargo-deny, cargo-machete]

set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ws-env.sh"

cargo fmt --all --check
cargo clippy --quiet --workspace --all-features --all-targets
cargo check --quiet --workspace --all-features
cargo doc --no-deps --all-features --package akuna-core --package akuna-ffi
cargo deny check --allow=no-license-field
cargo machete
