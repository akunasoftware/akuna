##### Nix Shell for CI Workspace Scripts & Local Development #####
{
  pkgs,
  mainPackage,
}:
let
  pname = mainPackage.pname;
  pVersion = mainPackage.version;
  rustVersion = mainPackage.passthru.rustVersion;
  rustToolChain = pkgs.rust-bin.stable.${rustVersion}.default.override {
    targets = [
      "aarch64-apple-darwin"
      "aarch64-unknown-linux-gnu"
      "x86_64-unknown-linux-gnu"
    ];
    extensions = [
      "clippy"
      "rust-src"
      "rust-analyzer"
    ];
  };

  aliases = [
    # Shorthand alias to run main package in OCI image
    (pkgs.writeShellScriptBin "akc" ''
       #!/usr/bin/env sh
       PNAME=$(basename $PWD)
       CONTAINER_NAME="akuna-dev-$PNAME"
       
      docker run -i --rm \
        --name "$CONTAINER_NAME" \
        -v "$PWD:/akuna" \
        -p "9876:9876" \
        akuna:v${pVersion} \
       "$@"
    '')

    # Shorthand alias for main package via cargo (use any time, but slower than debug out)
    (pkgs.writeShellScriptBin "akd" "cargo run -p ${pname} --all-features -- --log-level \"debug\" \"$@\"")

    # Shorthand alias for main package via debug out (must have already built)
    (pkgs.writeShellScriptBin "ak" "$PROJECT_ROOT/target/debug/${pname} \"$@\"")

    # Shorthand alias to build and open rustdoc for a specific crate in browser.
    # Usage: akdoc <crate-name>. Clears target/doc first; mirrors docs.rs:
    # all features enabled, deps excluded.
    (pkgs.writeShellScriptBin "akdoc" ''
      if [ $# -eq 0 ]; then
        echo "usage: akdoc <crate-name>" >&2
        exit 1
      fi
      rm -rf "$PROJECT_ROOT/target/doc"
      cargo doc --no-deps --all-features --open -p "$1"
    '')

    # Install main package to nix profile
    (pkgs.writeShellScriptBin "nix-install" ''
      set -euo pipefail

      target_system="''${NIX_TARGET_SYSTEM:-$(nix eval --impure --raw --expr 'builtins.currentSystem')}"

      cd "$PROJECT_ROOT"
      nix profile remove ${pname} || true
      nix profile add ".#packages.$target_system.default"
    '')

    # Show Nix closure size for main package output
    (pkgs.writeShellScriptBin "nix-inspect" ''
      set -euo pipefail
      system="''${NIX_TARGET_SYSTEM:-$(nix eval --impure --raw --expr 'builtins.currentSystem')}"
      while [ "$#" -gt 0 ]; do
        case "$1" in
          --system) system="$2"; shift 2 ;;
          *) echo "Unknown argument: $1" >&2; exit 1 ;;
        esac
      done
      out=$(nix build "$PROJECT_ROOT#packages.$system.default" --no-link --print-out-paths)
      nix path-info -rSh "$out"
    '')

    # Build OCI image and load into container runtime
    (pkgs.writeShellScriptBin "nix-oci-build" ''
      set -euo pipefail

      target_system="''${NIX_TARGET_SYSTEM:-$(nix eval --impure --raw --expr 'builtins.currentSystem')}"

      image_out="target/oci/image-$target_system.tar"
      mkdir -p "$(dirname "$image_out")"
      rm -f "$image_out"

      nix build ".#packages.$target_system.oci" -o "$image_out"
      docker load --input "$image_out"
    '')
  ];

in
pkgs.mkShell {

  buildInputs =
    with pkgs;
    [
      # core libs required by dev tools
      gcc
      zstd

      # general utils
      curl
      jq
      yq

      # services cli's
      python3Packages.huggingface-hub

      # language & framework tools
      rustToolChain
      uv # python project/test runner
      cargo-bloat # rust binary size inspection
      cargo-machete # rust dependency redundancy checker
      cargo-deny # rust dependency license checker
      sccache # rust compilation cache
    ]
    ++ aliases
    ++ mainPackage.passthru.dependencies.build;

  env = {
    LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
    RUSTC_WRAPPER = "${pkgs.sccache}/bin/sccache";
  }
  // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
    LIBRARY_PATH = mainPackage.passthru.env.runtime.LIBRARY_PATH;
    LD_LIBRARY_PATH = mainPackage.passthru.env.runtime.LD_LIBRARY_PATH;
    VK_DRIVER_FILES = mainPackage.passthru.env.runtime.VK_DRIVER_FILES;
  };

  shellHook = ''
    export PROJECT_ROOT=$(pwd);

    # set env using workspace env script (so it can still be used by non-nix users)
    . "$PROJECT_ROOT/build/scripts/ws-env.sh"
  '';

}
