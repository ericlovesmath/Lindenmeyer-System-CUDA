#pragma once

#include <nvtx3/nvToolsExt.h>

// NVTX_RANGE("name") opens a range that lasts until the enclosing scope ends.
struct nvtx_range {
  explicit nvtx_range(const char *name) { nvtxRangePushA(name); }
  ~nvtx_range() { nvtxRangePop(); }
  nvtx_range(const nvtx_range &) = delete;
  nvtx_range &operator=(const nvtx_range &) = delete;
};

#define NVTX_CONCAT_(a, b) a##b
#define NVTX_CONCAT(a, b) NVTX_CONCAT_(a, b)
#define NVTX_RANGE(name) nvtx_range NVTX_CONCAT(_nvtx_, __LINE__)(name)
