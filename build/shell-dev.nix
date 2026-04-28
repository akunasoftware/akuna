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
        localhost/akuna:${pVersion} \
       "$@"
    '')

    # Shorthand alias for main package via cargo (use any time, but slower than debug out)
    (pkgs.writeShellScriptBin "akd" "cargo run -p ${pname} -- --log-level \"debug\" \"$@\"")

    # Shorthand alias for main package via debug out (must have already built)
    (pkgs.writeShellScriptBin "ak" "$PROJECT_ROOT/target/debug/${pname} \"$@\"")

    # Install main package to nix profile
    (pkgs.writeShellScriptBin "nix-install" ''
      set -euo pipefail

      target_system="''${NIX_TARGET_SYSTEM:-$(nix eval --impure --raw --expr 'builtins.currentSystem')}"
      build_type="''${1:-release}"
      case "$build_type" in
        release) package_attr="default" ;;
        debug) package_attr="debug" ;;
        *) echo "Usage: nix-install [release|debug]" >&2; exit 1 ;;
      esac

      cd "$PROJECT_ROOT"
      nix profile remove ${pname} || true
      nix profile add ".#packages.$target_system.$package_attr"
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
      oci_system="''${NIX_OCI_SYSTEM:-x86_64-linux}"
      build_type="''${1:-release}"
      case "$build_type" in
        release) package_attr="oci-$oci_system" ;;
        debug) package_attr="oci-$oci_system-debug" ;;
        *) echo "Usage: nix-oci-build [release|debug]" >&2; exit 1 ;;
      esac

      image_out="target/oci/image-$oci_system-$build_type.tar"
      rm -f "$image_out"

      nix build ".#packages.$target_system.$package_attr" -o "$image_out"
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

      # general utils
      curl
      jq
      yq

      # language & framework tools
      rustToolChain
      cargo-bloat # rust binary size inspection
      cargo-nextest # rust test runner with nicer interface
      cargo-machete # rust dependency redundancy checker
      cargo-deny # rust dependency license checker
      sccache # rust compilation cache
      bacon # background rust code checker
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

    # if present, load sops-encrypted secrets into session env
    if [ -f "$PROJECT_ROOT/.env.enc" ]; then
      set -a
      source <(sops -d "$PROJECT_ROOT/.env.enc")
      set +a
    fi
  '';

}
