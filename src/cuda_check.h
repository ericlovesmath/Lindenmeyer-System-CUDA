#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

// Throws on a failed CUDA call
inline void check(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(err));
  }
}
