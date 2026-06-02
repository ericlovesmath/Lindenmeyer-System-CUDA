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

// Pooled device allocation: blocks freed by device_free() are kept and reused
void *pool_alloc(std::size_t bytes);

// Allocate `count` elements of T on the device, free with device_free()
template <typename T> T *device_alloc(std::size_t count) {
  return static_cast<T *>(pool_alloc(sizeof(T) * (count > 0 ? count : 1)));
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
