// Size-bucketed device allocation pool backing device_alloc()/device_free()

#include "gpu/cuda_check.h"    // pool_alloc declaration, check()
#include "gpu/device_buffer.h" // device_free declaration

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace {
// Freed device blocks kept for reuse, bucketed by exact byte size
std::unordered_map<std::size_t, std::vector<void *>> pool;
std::unordered_map<void *, std::size_t> block_size;
} // namespace

void *pool_alloc(std::size_t bytes) {
  auto &bin = pool[bytes];
  if (!bin.empty()) {
    void *p = bin.back();
    bin.pop_back();
    return p;
  }
  void *p = nullptr;
  check(cudaMalloc(&p, bytes), "cudaMalloc");
  block_size[p] = bytes;
  return p;
}

void device_free(void *p) noexcept {
  if (!p) {
    return;
  }
  auto it = block_size.find(p);
  if (it != block_size.end()) {
    pool[it->second].push_back(p);
  } else {
    cudaFree(p);
  }
}
