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
      cuda = pkgs.cudaPackages;
    in
    {
      devShells.${system}.default = (pkgs.mkShell.override { stdenv = cuda.backendStdenv; }) {
        packages = with pkgs; [
          cmake
          ninja
          git
          clang-tools
          virtualgl # vglrun to run GL to the real GPU over VNC

          cuda.cuda_nvcc
          cuda.cuda_cudart
          cuda.cuda_cccl # Thrust / CUB
          cuda.nsight_compute # `ncu` for `make profile`

          glfw
          libGL
        ];

        CUDA_PATH = cuda.cudatoolkit; # find_package(CUDAToolkit)

        # libcuda.so ships with the host driver, not nixpkgs
        shellHook = ''
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
        '';
      };
    };
}
