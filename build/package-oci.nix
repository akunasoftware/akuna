{
  pkgs,
  mainPackage,
  debug ? false,
}:
let
  pname = mainPackage.pname;
  pnameUpper = pkgs.lib.strings.toUpper pname;
  version = mainPackage.version;
  imageSuffix = pkgs.lib.optionalString debug "-debug";
  debugContents = [
    pkgs.bash
    pkgs.coreutils
    pkgs.lsof
  ]
  ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
    pkgs.iproute2
  ];
in
pkgs.dockerTools.buildLayeredImage {
  name = "${pname}${imageSuffix}";
  tag = "v${version}";

  contents = [
    mainPackage
    pkgs.cacert
  ]
  ++ pkgs.lib.optionals debug debugContents;

  config = {
    # by default, "run" main app command with info log level
    # user can run custom cmd instead which won't print info logs by default
    # default use case is expected to be running as a container service (worth printing info logs)
    Cmd = [
      "--log-level"
      "info"
      "serve"
    ];
    WorkingDir = "/${pname}";
    Entrypoint = [ "/bin/${pname}" ];
    Env = [
      # Application specific env vars, tells app where to look for config/data
      # Can be overriden by user config
      # Will not be used if local ".akuna" dir exists
      "${pnameUpper}_CONFIG_DIR=/${pname}/config"
      "${pnameUpper}_DATA_DIR=/${pname}/data"

      # General container env vars
      "HOME=/${pname}"
      "XDG_CONFIG_HOME=/${pname}/config"
      "XDG_DATA_HOME=/${pname}/data"
      "XDG_CACHE_HOME=/${pname}/.cache"

      # Nixpkg specific runtime vars
      "SSL_CERT_DIR=${pkgs.cacert}/etc/ssl/certs"
      "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
      "NIX_SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
    ];
  };
}
