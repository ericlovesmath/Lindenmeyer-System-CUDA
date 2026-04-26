{
  description = "Lindenmeyer System CUDA";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          cmake
          ninja
          gcc
          clang-tools
          cudaPackages.cuda_nvcc
          cudaPackages.cuda_cudart
        ];
      };
    };
}
