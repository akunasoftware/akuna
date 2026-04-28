#!/usr/bin/env bash

set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ws-env.sh"

cargo fmt --all
cargo fix --quiet --workspace --all-features --all-targets --allow-dirty --allow-staged
cargo clippy --quiet --fix --workspace --all-features --all-targets --allow-dirty --allow-staged
