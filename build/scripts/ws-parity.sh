#!/usr/bin/env bash
# Python FFI parity tests. [needs: cargo, uv]

set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ws-env.sh"

# Reject unsupported platforms before building anything.
case "$(uname -s)" in
Darwin) dylib="libakuna_ffi.dylib" ;;
Linux) dylib="libakuna_ffi.so" ;;
*)
	echo "unsupported platform for FFI parity: $(uname -s)" >&2
	exit 1
	;;
esac

bindings_dir="$PROJECT_ROOT/src-crates/ffi/tests/python/bindings"
library_path="$PROJECT_ROOT/target/debug/$dylib"

# Build the native library before replacing generated bindings.
cargo build -p akuna-ffi

# Regenerate bindings from the freshly built native library.
rm -rf "$bindings_dir"
mkdir -p "$bindings_dir"
cargo run --example uniffi-bindgen -p akuna-ffi -- "$library_path" "$bindings_dir"
cp "$library_path" "$bindings_dir/$dylib"

# Run cross-language parity after all native artifacts are ready.
uv run --project "$PROJECT_ROOT/src-crates/ffi/tests/python" pytest "$PROJECT_ROOT/src-crates/ffi/tests/python"
