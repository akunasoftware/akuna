#!/usr/bin/env bash

set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ws-env.sh"

./build/scripts/ws-fix.sh
./build/scripts/ws-check.sh
./build/scripts/ws-test.sh
