#!/usr/bin/env bash

set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ws-env.sh"

cargo fmt --all --check
cargo clippy --quiet --workspace --all-features --all-targets
cargo check --quiet --workspace --all-features
cargo machete
cargo deny check --allow=no-license-field
