# L-System on CUDA

GPU-accelerated [L-system](https://en.wikipedia.org/wiki/L-system) evaluation
and rendering of botanical structures. An L-system rewrites a string
of symbols in parallel under production rules, then a turtle interpreter walks
the result to draw self-similar fractal and plant-like forms.

## Running

Nix flake is provided to install dependencies, but the `CMakeList.txt` should be enough on CUDA systems

```bash
cmake -S . -B build
cmake --build build

# writes sample rendered ppm images to out/
make render

# runs tests
make test
```
