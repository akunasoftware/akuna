#!/usr/bin/env bash
# Static workspace checks. [needs: cargo +rustfmt +clippy, cargo-deny, cargo-machete, jq]

# Exit on errors, unset variables, or failed pipeline stages.
set -euo pipefail
# Load the shared workspace command environment.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ws-env.sh"

# Enforce Rust formatting.
cargo fmt --all --check

# Report dependencies unused by workspace source.
cargo machete

# Enforce dependency advisory, license, source, and duplication policy.
cargo deny check --allow=no-license-field

# Compile the always-on core surface without optional capabilities.
cargo check --quiet --package akuna-core --no-default-features

# Compile the complete workspace with every feature enabled.
cargo check --quiet --workspace --all-features

# Lint every workspace target with every feature enabled.
cargo clippy --quiet --workspace --all-features --all-targets

# Build public API documentation for core and FFI.
cargo doc --no-deps --all-features --package akuna-core --package akuna-ffi

# Compile every capability in `full` alone to catch undeclared feature coupling.
cargo metadata --no-deps --format-version 1 |
	jq -r '.packages[] | select(.name == "akuna-core") | .features.full[]' |
	while IFS= read -r feature; do
		cargo check --quiet --package akuna-core --no-default-features --features "$feature"
	done
