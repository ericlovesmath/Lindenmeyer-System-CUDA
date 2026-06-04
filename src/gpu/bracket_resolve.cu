// On-device resolver for bracketed turtle systems

#include "gpu/turtle_engine.h"

#include "gpu/kernel.h"
#include "gpu/nvtx.h"

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>

// The resolver's reusable device buffers, kept off the shared header
struct bracket_scratch {
  device_buffer<int> depth, qidx, openpos, branch, skey, order, jumpA, jumpB;
  device_buffer<long long> qkey, openkey;
  device_buffer<frame3> sorted, prefix, accA, accB;

  int resolve_branches(const device_buffer<unsigned char> &commands);
  void resolve_prefix(const device_buffer<unsigned char> &commands,
                      const turtle_config &cfg);
  frame3 *resolve_entries(const device_buffer<unsigned char> &commands,
                          int maxdepth);
};

namespace {

constexpr int ROOT = -1;
constexpr frame3 IDENTITY{{1.0, 0.0, 0.0, 0.0}, 0.0, 0.0, 0.0};

struct is_open {
  __host__ __device__ bool operator()(unsigned char c) const {
    return c == '[';
  }
};
struct to_delta { // signed bracket-depth contribution of a symbol
  __host__ __device__ int operator()(unsigned char c) const {
    return c == '[' ? 1 : (c == ']' ? -1 : 0);
  }
};

// A '[' belongs to its parent branch, so it looks up one level shallower.
__host__ __device__ int level_at(unsigned char c, int depth) {
  return c == '[' ? depth - 1 : depth;
}
// (level, position) packed so a sort/search orders by level then position.
__host__ __device__ long long key_of(int level, int pos, long long n1) {
  return static_cast<long long>(level) * n1 + pos;
}

// Pack each symbol's lookup key (a '[' belongs to its parent, one level up).
__global__ void key_kernel(int n, const unsigned char *cmd, const int *depth,
                           long long n1, long long *qkey) {
  GRID_STRIDE_LOOP(i, n)
  qkey[i] = key_of(level_at(cmd[i], depth[i]), i, n1);
}

// Pack the key of each '[' so the open list can be sorted and searched.
__global__ void openkey_kernel(int nopen, const int *openpos, const int *depth,
                               long long n1, long long *openkey) {
  GRID_STRIDE_LOOP(k, nopen)
  openkey[k] = key_of(depth[openpos[k]], openpos[k], n1);
}

// branch[i] = innermost enclosing '['
__global__ void branch_kernel(int n, const unsigned char *cmd, const int *depth,
                              const int *qidx, const long long *openkey,
                              const int *openpos, long long n1, int *branch) {
  GRID_STRIDE_LOOP(i, n) {
    int j = qidx[i];
    branch[i] = (j > 0 && openkey[j - 1] / n1 == level_at(cmd[i], depth[i]))
                    ? openpos[j - 1]
                    : ROOT;
  }
}

__global__ void doubling_kernel(int n, const unsigned char *cmd,
                                const frame3 *accIn, const int *jumpIn,
                                frame3 *accOut, int *jumpOut) {
  compose comp;
  GRID_STRIDE_LOOP(i, n) {
    int p = cmd[i] == '[' ? jumpIn[i] : ROOT;
    accOut[i] = p < 0 ? accIn[i] : comp(accIn[p], accIn[i]);
    jumpOut[i] = p < 0 ? ROOT : jumpIn[p];
  }
}

__global__ void world_b_kernel(int n, const int *branch, const frame3 *acc,
                               const frame3 *P, frame3 init, frame3 *world) {
  compose comp;
  GRID_STRIDE_LOOP(i, n) {
    int b = branch[i];
    world[i] = comp(b < 0 ? init : comp(init, acc[b]), P[i]);
  }
}

} // namespace

// branch[i] = innermost enclosing '['. Returns the bracket tree's height.
int bracket_scratch::resolve_branches(
    const device_buffer<unsigned char> &commands) {
  NVTX_RANGE("brk-branches");
  const int n = commands.size;
  const long long n1 = static_cast<long long>(n) + 1;
  auto cmd = dptr(commands.data);

  int *dep = grow(depth, n);
  auto deltas = thrust::make_transform_iterator(cmd, to_delta{});
  thrust::inclusive_scan(par, deltas, deltas + n, dptr(dep));
  int maxdepth =
      thrust::reduce(par, dptr(dep), dptr(dep) + n, 0, thrust::maximum<int>());

  launch("key", key_kernel, n, n, commands.data, dep, n1, grow(qkey, n));

  int *op = grow(openpos, n);
  int nopen =
      static_cast<int>(thrust::copy_if(par, thrust::make_counting_iterator(0),
                                       thrust::make_counting_iterator(n), cmd,
                                       dptr(op), is_open{}) -
                       dptr(op));
  long long *ok = grow(openkey, n);
  launch("openkey", openkey_kernel, nopen, nopen, op, dep, n1, ok);
  thrust::sort_by_key(par, dptr(ok), dptr(ok) + nopen, dptr(op));

  thrust::lower_bound(par, dptr(ok), dptr(ok) + nopen, dptr(qkey.data),
                      dptr(qkey.data) + n, dptr(grow(qidx, n)));
  launch("branch", branch_kernel, n, n, commands.data, dep, qidx.data, ok, op,
         n1, grow(branch, n));
  return maxdepth;
}

// prefix[i] = the within-branch exclusive compose-scan of the local transforms.
void bracket_scratch::resolve_prefix(
    const device_buffer<unsigned char> &commands, const turtle_config &cfg) {
  NVTX_RANGE("brk-prefix");
  const int n = commands.size;
  auto locals = thrust::make_transform_iterator(
      dptr(commands.data), to_local{cfg.step, radians(cfg.angle_deg)});

  int *sk = grow(skey, n), *od = grow(order, n);
  thrust::copy(par, dptr(branch.data), dptr(branch.data) + n, dptr(sk));
  thrust::sequence(par, dptr(od), dptr(od) + n);
  thrust::stable_sort_by_key(par, dptr(sk), dptr(sk) + n, dptr(od));

  frame3 *sr = grow(sorted, n);
  thrust::gather(par, dptr(od), dptr(od) + n, locals, dptr(sr));
  thrust::exclusive_scan_by_key(par, dptr(sk), dptr(sk) + n, dptr(sr), dptr(sr),
                                IDENTITY, thrust::equal_to<int>(), compose{});
  thrust::scatter(par, dptr(sr), dptr(sr) + n, dptr(od), dptr(grow(prefix, n)));
}

// Carry each '[' entry transform up the tree by pointer doubling over
// parent = branch[b], seeding each '[' with its own prefix.
frame3 *
bracket_scratch::resolve_entries(const device_buffer<unsigned char> &commands,
                                 int maxdepth) {
  NVTX_RANGE("brk-entries");
  const int n = commands.size;
  frame3 *a = grow(accA, n), *b = grow(accB, n);
  int *ja = grow(jumpA, n), *jb = grow(jumpB, n);
  thrust::copy(par, dptr(prefix.data), dptr(prefix.data) + n, dptr(a));
  thrust::copy(par, dptr(branch.data), dptr(branch.data) + n, dptr(ja));
  for (int step = 1; step < maxdepth; step <<= 1) { // chain length <= maxdepth
    launch("double", doubling_kernel, n, n, commands.data, a, ja, b, jb);
    std::swap(a, b);
    std::swap(ja, jb);
  }
  return a;
}

turtle_engine::~turtle_engine() {
  if (host)
    cudaFreeHost(host);
  delete brk;
}

void turtle_engine::fill_world_bracketed(
    const device_buffer<unsigned char> &commands, const turtle_config &cfg,
    frame3 *world) {
  NVTX_RANGE("brk-world");
  const int n = commands.size;
  if (n <= 0)
    return;
  if (!brk)
    brk = new bracket_scratch;

  const int maxdepth = brk->resolve_branches(commands);
  brk->resolve_prefix(commands, cfg);
  const frame3 *entry = brk->resolve_entries(commands, maxdepth);

  const frame3 init{quat_axis_angle(up_axis(), radians(cfg.start_heading_deg)),
                    0.0, 0.0, 0.0};
  launch("world_b", world_b_kernel, n, n, brk->branch.data, entry,
         brk->prefix.data, init, world);
  check(cudaDeviceSynchronize(), "fill_world_bracketed");
}
