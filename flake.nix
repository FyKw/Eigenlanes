{
  description = "Lane compression comparisson CLI tool";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; };
  in {
    packages.${system}.default = pkgs.python312.withPackages (ps: with ps; [
    pip
    ]);

    apps.${system}.default = {
      type = "app";
      program = "${self.packages.${system}.default}/bin/python";
      args = [ "-m" "src" ];
    };

    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        ffmpeg
        self.packages.${system}.default
      ];
    };
  };
}
