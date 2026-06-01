#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

// Throws on a failed CUDA call
inline void check(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(err));
  }
}

// Allocate `count` elements of T on the device (at least one). Free with
// device_free() or cudaFree().
template <typename T> T *device_alloc(std::size_t count) {
  T *p = nullptr;
  check(cudaMalloc(&p, sizeof(T) * (count > 0 ? count : 1)), "cudaMalloc");
  return p;
}

// Copy `count` elements host to device and vice versa
template <typename T> void upload(T *dst, const T *src, std::size_t count) {
  if (count > 0) {
    check(cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice),
          "cudaMemcpy H2D");
  }
}
template <typename T> void download(T *dst, const T *src, std::size_t count) {
  if (count > 0) {
    check(cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H");
  }
}
