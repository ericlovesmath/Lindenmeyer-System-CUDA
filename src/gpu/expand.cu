#include "gpu/expand.h"

#include "gpu/cuda_check.h"
#include "gpu/kernel.h"
#include "gpu/nvtx.h"

#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <array>
#include <string>
#include <vector>

namespace {

constexpr int ALPHABET = 256;

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
  GRID_STRIDE_LOOP(i, n) { len[i] = rule_len[in[i]]; }
}

// Copy each input symbol's replacement into its scanned output offset
__global__ void scatter_kernel(const unsigned char *in, int n,
                               const int *rule_off, const int *rule_len,
                               const char *rule_data, const int *offset,
                               unsigned char *out) {
  GRID_STRIDE_LOOP(i, n) {
    unsigned char c = in[i];
    const char *src = rule_data + rule_off[c];
    unsigned char *dst = out + offset[i];
    int l = rule_len[c];
    for (int k = 0; k < l; ++k) {
      dst[k] = src[k];
    }
  }
}

// Rewrite `in` (n symbols) once into a new device buffer (caller owns both).
// The new length is written to *out_n.
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

  device_free(len);
  device_free(off);
  *out_n = total;
  return out;
}

} // namespace

device_buffer<unsigned char> to_device(const std::string &s) {
  const int n = static_cast<int>(s.size());
  unsigned char *d = device_alloc<unsigned char>(n);
  upload(d, reinterpret_cast<const unsigned char *>(s.data()), n);
  return {d, n};
}

std::string to_host(const device_buffer<unsigned char> &buf) {
  std::string out(buf.size, '\0');
  download(reinterpret_cast<unsigned char *>(out.data()), buf.data, buf.size);
  return out;
}

device_buffer<unsigned char> expand_device(const l_system &sys,
                                           int iterations) {
  rule_table t = build_rule_table(sys);

  // Upload the rule table and axiom, then rewrite in place on the device.
  device_rules rules{device_alloc<int>(ALPHABET), device_alloc<int>(ALPHABET),
                     device_alloc<char>(t.data.size())};
  int n = static_cast<int>(sys.axiom.size());
  unsigned char *cur = device_alloc<unsigned char>(n);
  upload(rules.off, t.off.data(), ALPHABET);
  upload(rules.len, t.len.data(), ALPHABET);
  upload(rules.data, t.data.data(), t.data.size());
  upload(cur, reinterpret_cast<const unsigned char *>(sys.axiom.data()), n);

  // Each step hands back a freshly sized buffer.
  for (int it = 0; it < iterations && n > 0; ++it) {
    NVTX_RANGE("expand-step");
    int next_n = 0;
    unsigned char *next = expand_step(cur, n, rules, &next_n);
    device_free(cur);
    cur = next;
    n = next_n;
  }

  device_free(rules.off);
  device_free(rules.len);
  device_free(rules.data);
  check(cudaDeviceSynchronize(), "sync"); // result is ready when we return
  return {cur, n};
}

std::string expand_gpu(const l_system &sys, int iterations) {
  return to_host(expand_device(sys, iterations));
}
