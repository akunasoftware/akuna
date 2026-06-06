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
          guestSystemFor = system: lib.replaceStrings [ "darwin" ] [ "linux" ] system;
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
          mkOciPackage =
            targetSystem:
            let
              ociPkgs = mkPkgs targetSystem;
              ociMainPackage = ociPkgs.callPackage ./build/package-main.nix { };
            in
            ociPkgs.callPackage ./build/package-oci.nix {
              mainPackage = ociMainPackage;
            };
        in
        {
          packages = {
            default = mainPackage;
            oci = mkOciPackage (guestSystemFor system);
          };

          devShells.default = import ./build/shell-dev.nix {
            inherit pkgs mainPackage;
          };
        }
      );
}
