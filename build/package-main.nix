{
  lib,
  pkgs,
}:
let
  pname = "akuna";
  cargoPackageName = "akuna";
  version = (lib.importTOML ../Cargo.toml).workspace.package.version;
  rustVersion = "1.96.0";
  rustToolChain = pkgs.rust-bin.stable.${rustVersion}.minimal.override {
    targets = [
      "aarch64-apple-darwin"
      "aarch64-unknown-linux-gnu"
      "x86_64-unknown-linux-gnu"
    ];
  };

  # Deps only required while building or in devshells
  nativeBuildDependencies = [
    pkgs.pkg-config
    pkgs.clang
    pkgs.cmake
    pkgs.perl
    pkgs.cacert
  ];

  # Deps always required at runtime on nix systems
  runDependencies = [
    pkgs.openssl
  ]
  ++ lib.optionals pkgs.stdenv.isLinux [
    pkgs.vulkan-loader
  ];

  runtimeLibraryPath = lib.makeLibraryPath runDependencies;
  runtimePkgConfigPath = lib.makeSearchPathOutput "dev" "lib/pkgconfig" runDependencies;

  # Nix package runtime environment vars (do not use in devshell)
  runtimeEnv = {
    PKG_CONFIG_PATH = runtimePkgConfigPath;
  }
  // lib.optionalAttrs pkgs.stdenv.isLinux {
    LD_LIBRARY_PATH = lib.concatStringsSep ":" [
      runtimeLibraryPath
      "/run/opengl-driver/lib"
    ];
    LIBRARY_PATH = runtimeLibraryPath;
    VK_DRIVER_FILES = "/run/opengl-driver/share/vulkan/icd.d";
  }
  // lib.optionalAttrs pkgs.stdenv.isDarwin {
    DYLD_LIBRARY_PATH = runtimeLibraryPath;
    DYLD_FALLBACK_LIBRARY_PATH = runtimeLibraryPath;
    LIBRARY_PATH = runtimeLibraryPath;
  };

  # Nix package build environment vars (do not use in devshell)
  buildEnv = {
    LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
    SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
    CARGO_HTTP_CAINFO = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
    NIX_SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
    OPENSSL_NO_VENDOR = "1";
    OPENSSL_STATIC = "0";
    OPENSSL_DIR = "${pkgs.openssl.dev}";
    OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
    OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
  }
  // lib.optionalAttrs pkgs.stdenv.isDarwin {
    NIX_LDFLAGS = "-dead_strip_dylibs";
  };

  rustPlatform = pkgs.makeRustPlatform {
    cargo = rustToolChain;
    rustc = rustToolChain;
  };
in
rustPlatform.buildRustPackage {
  pname = pname;
  version = version;
  src = ../.;
  env = buildEnv // runtimeEnv;
  nativeBuildInputs = nativeBuildDependencies;
  buildInputs = runDependencies;

  doCheck = false;
  enableParallelBuilding = true;

  # git-sourced dependencies require explicit hashes
  cargoLock = {
    lockFile = ../Cargo.lock;
  };

  # Only build the main crate, not all workspace members
  cargoBuildFlags = [
    "--package=${cargoPackageName}"
  ];

  postInstall = lib.optionalString pkgs.stdenv.isLinux ''
    strip --strip-unneeded "$out/bin/${pname}"
  '';

  meta = {
    description = "Akuna Knowledge Tools";
    homepage = "https://akuna.software";
    downloadPage = "https://github.com/akunasoftware/akuna";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [
      smissingham
    ];
    platforms = lib.platforms.linux ++ lib.platforms.darwin;
  };

  passthru = {
    rustVersion = rustVersion;
    dependencies = {
      build = nativeBuildDependencies;
      runtime = runDependencies;
    };
    env = {
      build = buildEnv;
      runtime = runtimeEnv;
    };
  };
}
