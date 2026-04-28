{
  description = "Nix flake for project development and packaging";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      rust-overlay,
      ...
    }:
    flake-utils.lib.eachSystem
      [
        "aarch64-linux"
        "x86_64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ]
      (
        system:
        let
          lib = nixpkgs.lib;
          linuxOciSystems = [
            "x86_64-linux"
            "aarch64-linux"
          ];
          overlays = [ (import rust-overlay) ];
          mkPkgs =
            targetSystem:
            import nixpkgs {
              config = {
                allowUnfree = true;
                allowUnsupportedSystem = false;
              };
              inherit overlays;
              system = targetSystem;
            };
          pkgs = import nixpkgs {
            config = {
              allowUnfree = true;
              allowUnsupportedSystem = false;
            };
            inherit system overlays;
          };
          mainPackage = pkgs.callPackage ./build/package-main.nix { };
          debugPackage = pkgs.callPackage ./build/package-main.nix { buildType = "debug"; };
          mkOciPackage =
            targetSystem: debug:
            let
              ociPkgs = mkPkgs targetSystem;
              ociMainPackage = ociPkgs.callPackage ./build/package-main.nix {
                buildType = if debug then "debug" else "release";
              };
            in
            ociPkgs.callPackage ./build/package-oci.nix {
              mainPackage = ociMainPackage;
              inherit debug;
            };
          ociPackages = lib.genAttrs linuxOciSystems (
            targetSystem: mkOciPackage targetSystem false
          );
          ociDebugPackages = lib.genAttrs linuxOciSystems (
            targetSystem: mkOciPackage targetSystem true
          );
        in
        {
          packages = {
            default = mainPackage;
            debug = debugPackage;
          }
          // lib.mapAttrs' (targetSystem: package: lib.nameValuePair "oci-${targetSystem}" package) ociPackages
          // lib.mapAttrs'
            (targetSystem: package: lib.nameValuePair "oci-${targetSystem}-debug" package)
            ociDebugPackages;

          checks = {
            default = mainPackage;
            debug = debugPackage;
          };

          devShells.default = import ./build/shell-dev.nix {
            inherit pkgs mainPackage;
          };
        }
      );
}
