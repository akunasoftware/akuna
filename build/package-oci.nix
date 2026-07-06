{
  pkgs,
  mainPackage,
  debug ? false,
}:
let
  pname = mainPackage.pname;
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
    WorkingDir = "/${pname}";
    Entrypoint = [ "/bin/${pname}" ];
    Env = [
      # General container env vars
      "HOME=/${pname}"
      "XDG_CACHE_HOME=/${pname}/.cache"

      # Nixpkg specific runtime vars
      "SSL_CERT_DIR=${pkgs.cacert}/etc/ssl/certs"
      "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
      "NIX_SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
    ];
  };
}
