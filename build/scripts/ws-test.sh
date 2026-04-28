#!/usr/bin/env bash

set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ws-env.sh"

cargo nextest run --all-features
