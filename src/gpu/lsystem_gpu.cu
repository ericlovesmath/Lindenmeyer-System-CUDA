#include "gpu/lsystem_gpu.h"

#include "gpu/cuda_check.h"

#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <array>
#include <string>
#include <vector>

namespace {

constexpr int ALPHABET = 256;
constexpr int BLOCK = 256;

int grid_for(int n) { return (n + BLOCK - 1) / BLOCK; }

// Allocate `count` of `T` elements on the device, user calls `cudaFree`
template <typename T> T *device_alloc(std::size_t count) {
  T *p = nullptr;
  check(cudaMalloc(&p, sizeof(T) * (count > 0 ? count : 1)), "cudaMalloc");
  return p;
}

// Copies host to to device
template <typename T> void upload(T *dst, const T *src, std::size_t count) {
  if (count > 0) {
    check(cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice),
          "cudaMemcpy H2D");
  }
}

// Copies device to host
template <typename T> void download(T *dst, const T *src, std::size_t count) {
  if (count > 0) {
    check(cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H");
  }
}

// Launch a grid-strided kernel over `n` items and check
template <typename Kernel, typename... Args>
void launch(const char *name, Kernel kernel, int n, Args... args) {
  kernel<<<grid_for(n), BLOCK>>>(args...);
  check(cudaGetLastError(), name);
}

// Flattened, branch-free production table
struct rule_table {
  std::vector<char> data;          // all replacements concatenated
  std::array<int, ALPHABET> off{}; // start of each symbol's replacement
  std::array<int, ALPHABET> len{}; // length of each symbol's replacement
};

rule_table build_rule_table(const l_system &sys) {
  rule_table t;
  // Default is just identity
  for (int c = 0; c < ALPHABET; ++c) {
    t.off[c] = static_cast<int>(t.data.size());
    t.len[c] = 1;
    t.data.push_back(static_cast<char>(c));
  }
  // Override with the grammar's productions.
  for (const auto &rule : sys.rules) {
    unsigned char c = static_cast<unsigned char>(rule.first);
    const std::string &rhs = rule.second;
    t.off[c] = static_cast<int>(t.data.size());
    t.len[c] = static_cast<int>(rhs.size());
    t.data.insert(t.data.end(), rhs.begin(), rhs.end());
  }
  return t;
}

// The rule table living on the device
struct device_rules {
  int *off;
  int *len;
  char *data;
};

// len[i] = replacement length of input symbol i
__global__ void length_kernel(const unsigned char *in, int n,
                              const int *rule_len, int *len) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    len[i] = rule_len[in[i]];
  }
}

// Copy each input symbol's replacement into its scanned output offset
__global__ void scatter_kernel(const unsigned char *in, int n,
                               const int *rule_off, const int *rule_len,
                               const char *rule_data, const int *offset,
                               unsigned char *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    unsigned char c = in[i];
    const char *src = rule_data + rule_off[c];
    unsigned char *dst = out + offset[i];
    int l = rule_len[c];
    for (int k = 0; k < l; ++k) {
      dst[k] = src[k];
    }
  }
}

// Rewrite `in` (n symbols) once and return the new device buffer
// New length is written to *out_n
// The caller still owns `in`
unsigned char *expand_step(const unsigned char *in, int n,
                           const device_rules &rules, int *out_n) {
  int *len = device_alloc<int>(n);
  int *off = device_alloc<int>(n);

  launch("length_kernel", length_kernel, n, in, n, rules.len, len);

  // This gives us per-symbol lengths gives each symbol's write offset
  thrust::exclusive_scan(thrust::device, len, len + n, off);

  // Output size = offset of the last symbol + its replacement length.
  int last_off = 0, last_len = 0;
  download(&last_off, off + (n - 1), 1);
  download(&last_len, len + (n - 1), 1);
  int total = last_off + last_len;

  unsigned char *out = device_alloc<unsigned char>(total);
  launch("scatter_kernel", scatter_kernel, n, in, n, rules.off, rules.len,
         rules.data, off, out);

  cudaFree(len);
  cudaFree(off);
  *out_n = total;
  return out;
}

} // namespace

std::string expand_gpu(const l_system &sys, int iterations) {
  rule_table t = build_rule_table(sys);

  // Upload the rule table once
  device_rules rules{device_alloc<int>(ALPHABET), device_alloc<int>(ALPHABET),
                     device_alloc<char>(t.data.size())};
  upload(rules.off, t.off.data(), ALPHABET);
  upload(rules.len, t.len.data(), ALPHABET);
  upload(rules.data, t.data.data(), t.data.size());

  // Current string lives in `cur`, each step hands back a freshly sized buffer
  int n = static_cast<int>(sys.axiom.size());
  unsigned char *cur = device_alloc<unsigned char>(n);
  upload(cur, reinterpret_cast<const unsigned char *>(sys.axiom.data()), n);

  for (int it = 0; it < iterations && n > 0; ++it) {
    int next_n = 0;
    unsigned char *next = expand_step(cur, n, rules, &next_n);
    cudaFree(cur);
    cur = next;
    n = next_n;
  }

  check(cudaDeviceSynchronize(), "sync");

  std::string result(n, '\0');
  download(reinterpret_cast<unsigned char *>(result.data()), cur, n);

  cudaFree(cur);
  cudaFree(rules.off);
  cudaFree(rules.len);
  cudaFree(rules.data);
  return result;
}
