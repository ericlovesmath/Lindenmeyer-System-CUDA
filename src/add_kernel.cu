#include "add_kernel.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace {

__global__ void add_kernel(const int *a, const int *b, int *out) {
  *out = *a + *b;
}

void check(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(err));
  }
}

} // namespace

int add_with_cuda(int a, int b) {
  int *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  check(cudaMalloc(&d_a, sizeof(int)), "cudaMalloc d_a");
  check(cudaMalloc(&d_b, sizeof(int)), "cudaMalloc d_b");
  check(cudaMalloc(&d_out, sizeof(int)), "cudaMalloc d_out");

  check(cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice), "memcpy a");
  check(cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice), "memcpy b");

  add_kernel<<<1, 1>>>(d_a, d_b, d_out);
  check(cudaGetLastError(), "kernel launch");
  check(cudaDeviceSynchronize(), "sync");

  int out = 0;

  check(cudaMemcpy(&out, d_out, sizeof(int), cudaMemcpyDeviceToHost),
        "memcpy out");

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
  return out;
}
