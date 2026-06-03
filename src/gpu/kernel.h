#pragma once

// Shared CUDA kernel-launch primitives used by the GPU library.

#include "gpu/cuda_check.h"

#include <cuda_runtime.h>

// Threads per block for every kernel launch below.
constexpr int BLOCK = 256;

// Blocks needed to cover `n` items at BLOCK threads each.
inline int grid_for(int n) { return (n + BLOCK - 1) / BLOCK; }

// Launch a grid-strided kernel over `n` items and check for launch errors.
// A non-positive `n` is a no-op (some stages legitimately get empty input).
template <typename Kernel, typename... Args>
void launch(const char *name, Kernel kernel, int n, Args... args) {
  if (n <= 0)
    return;
  kernel<<<grid_for(n), BLOCK>>>(args...);
  check(cudaGetLastError(), name);
}

// Every kernel walks its range with this grid-stride loop.
#define GRID_STRIDE_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)
