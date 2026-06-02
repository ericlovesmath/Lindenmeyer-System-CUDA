#include "gpu/interpret.h"

#include "gpu/cuda_check.h"  // check(), device_alloc(), download()
#include "gpu/expand.h" // to_device() for the host-string convenience

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#include <algorithm>

#include <cfloat>
#include <cmath>
#include <string>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;

constexpr double radians(double degrees) { return degrees * PI / 180.0; }

constexpr int BLOCK = 256;
int grid_for(int n) { return (n + BLOCK - 1) / BLOCK; }

// Launch a grid-strided kernel over `n` items and check
template <typename Kernel, typename... Args>
void launch(const char *name, Kernel kernel, int n, Args... args) {
  if (n <= 0)
    return;
  kernel<<<grid_for(n), BLOCK>>>(args...);
  check(cudaGetLastError(), name);
}

// Every kernel below walks its range with this grid-stride loop.
#define GRID_STRIDE_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

// Shorthands used throughout: a device_ptr cast and the async execution policy.
template <class T> thrust::device_ptr<T> dptr(T *p) {
  return thrust::device_pointer_cast(p);
}
const auto par = thrust::cuda::par_nosync;

// A 2D rigid transform acting on a point
struct affine2 {
  double c, s, tx, ty;
};

// Compose two transforms
struct compose {
  __host__ __device__ affine2 operator()(const affine2 &a,
                                         const affine2 &b) const {
    return {a.c * b.c - a.s * b.s, a.s * b.c + a.c * b.s,
            a.c * b.tx - a.s * b.ty + a.tx, a.s * b.tx + a.c * b.ty + a.ty};
  }
};

// Map a symbol to its local transform
struct to_local {
  double step, cos_d, sin_d;
  __host__ __device__ affine2 operator()(unsigned char ch) const {
    switch (ch) {
    case 'F':
    case 'f':
      return {1.0, 0.0, step, 0.0};
    case '+':
      return {cos_d, sin_d, 0.0, 0.0};
    case '-':
      return {cos_d, -sin_d, 0.0, 0.0};
    default:
      return {1.0, 0.0, 0.0, 0.0}; // identity
    }
  }
};

// Turn a turtle frame into the segment a forward step would draw from it
struct to_segment {
  double step;
  __host__ __device__ segment operator()(const affine2 &w) const {
    return {{w.tx, w.ty}, {w.tx + step * w.c, w.ty + step * w.s}};
  }
};

// Narrow a turtle frame to a float OpenGL instance attribute
struct to_frame {
  __host__ __device__ gpu_frame operator()(const affine2 &w) const {
    return {static_cast<float>(w.tx), static_cast<float>(w.ty),
            static_cast<float>(w.c), static_cast<float>(w.s)};
  }
};

struct is_forward {
  __host__ __device__ bool operator()(unsigned char ch) const {
    return ch == 'F';
  }
};

void scan_world(const device_buffer<unsigned char> &commands,
                const turtle_config &cfg, affine2 *world) {
  const int n = commands.size;
  const double delta = radians(cfg.angle_deg),
               h = radians(cfg.start_heading_deg);
  const to_local local{cfg.step, std::cos(delta), std::sin(delta)};
  const affine2 init{std::cos(h), std::sin(h), 0.0, 0.0};
  auto locals = thrust::make_transform_iterator(dptr(commands.data), local);
  thrust::exclusive_scan(par, locals, locals + n, dptr(world), init, compose{});
}

gpu_frame *pinned_alloc(int count) {
  gpu_frame *p = nullptr;
  check(cudaHostAlloc(&p, sizeof(gpu_frame) * (count > 0 ? count : 1),
                      cudaHostAllocDefault),
        "cudaHostAlloc");
  return p;
}
void pinned_free(gpu_frame *p) noexcept {
  if (p) {
    cudaFreeHost(p);
  }
}

// Reallocate `b` only if it must grow
template <class T> T *grow(device_buffer<T> &b, int n) {
  if (b.size < n)
    b = device_buffer<T>(device_alloc<T>(n), n);
  return b.data;
}

// Scratch reused across calls so a recompute allocates nothing
struct scratch {
  device_buffer<affine2> world;    // scan output
  device_buffer<gpu_frame> frames; // compacted F frames (fallback)
  gpu_frame *host = nullptr;       // pinned D2H staging (fallback)
  int host_cap = 0;

  // Bracketed-turtle temporaries (see scan_world_bracketed)
  device_buffer<int> depth;          // bracket nesting depth
  device_buffer<long long> qkey;     // per-symbol (level, position) lookup key
  device_buffer<int> qidx;           // lower_bound of qkey in the open keys
  device_buffer<int> openpos;        // '[' positions, sorted by openkey
  device_buffer<long long> openkey;  // (level, position) key per '['
  device_buffer<int> branch;         // innermost enclosing '[' (ROOT = -1)
  device_buffer<int> skey, order;    // branch ids + permutation that sorts them
  device_buffer<affine2> sorted;     // locals in branch order, scanned in place
  device_buffer<affine2> prefix;     // within-branch prefix P, original order
  device_buffer<affine2> accA, accB; // pointer-doubling entry accumulator
  device_buffer<int> jumpA, jumpB;   // pointer-doubling parent jump

  ~scratch() { pinned_free(host); }

  gpu_frame *host_ptr(int n) {
    if (host_cap < n) {
      pinned_free(host);
      host = pinned_alloc(n);
      host_cap = n;
    }
    return host;
  }
};
scratch g;

// Bounding box of the segment a frame draws (both endpoints).
struct frame_extent {
  float step;
  __host__ __device__ bounds2 operator()(const gpu_frame &f) const {
    float ex = f.tx + step * f.c, ey = f.ty + step * f.s;
    return {fminf(f.tx, ex), fminf(f.ty, ey), fmaxf(f.tx, ex), fmaxf(f.ty, ey)};
  }
};

struct bounds_union {
  __host__ __device__ bounds2 operator()(const bounds2 &a,
                                         const bounds2 &b) const {
    return {fminf(a.min_x, b.min_x), fminf(a.min_y, b.min_y),
            fmaxf(a.max_x, b.max_x), fmaxf(a.max_y, b.max_y)};
  }
};

// Bounding box of the geometry `count` frames draw (a GPU reduction).
bounds2 frames_bounds(const gpu_frame *frames, int count, float step) {
  if (count == 0)
    return {0.0f, 0.0f, 0.0f, 0.0f};
  const bounds2 init{FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX};
  auto p = thrust::device_pointer_cast(frames);
  return thrust::transform_reduce(thrust::cuda::par, p, p + count,
                                  frame_extent{step}, init, bounds_union{});
}

constexpr int ROOT = -1;
constexpr affine2 IDENTITY{1.0, 0.0, 0.0, 0.0};

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
                                const affine2 *accIn, const int *jumpIn,
                                affine2 *accOut, int *jumpOut) {
  compose comp;
  GRID_STRIDE_LOOP(i, n) {
    int p = cmd[i] == '[' ? jumpIn[i] : ROOT;
    accOut[i] = p < 0 ? accIn[i] : comp(accIn[p], accIn[i]);
    jumpOut[i] = p < 0 ? ROOT : jumpIn[p];
  }
}

__global__ void world_b_kernel(int n, const int *branch, const affine2 *acc,
                               const affine2 *P, affine2 init, affine2 *world) {
  compose comp;
  GRID_STRIDE_LOOP(i, n) {
    int b = branch[i];
    world[i] = comp(b < 0 ? init : comp(init, acc[b]), P[i]);
  }
}

// Step 1. Fill g.branch[i] = innermost enclosing '['
// Returns the max nesting depth (height of the bracket tree)
int resolve_branches(const device_buffer<unsigned char> &commands) {
  const int n = commands.size;
  const long long n1 = static_cast<long long>(n) + 1;
  auto cmd = dptr(commands.data);

  int *depth = grow(g.depth, n);
  auto deltas = thrust::make_transform_iterator(cmd, to_delta{});
  thrust::inclusive_scan(par, deltas, deltas + n, dptr(depth));
  int maxdepth = thrust::reduce(par, dptr(depth), dptr(depth) + n, 0,
                                thrust::maximum<int>());

  launch("key", key_kernel, n, n, commands.data, depth, n1, grow(g.qkey, n));

  int *openpos = grow(g.openpos, n);
  int nopen =
      static_cast<int>(thrust::copy_if(par, thrust::make_counting_iterator(0),
                                       thrust::make_counting_iterator(n), cmd,
                                       dptr(openpos), is_open{}) -
                       dptr(openpos));
  long long *openkey = grow(g.openkey, n);
  launch("openkey", openkey_kernel, nopen, nopen, openpos, depth, n1, openkey);
  thrust::sort_by_key(par, dptr(openkey), dptr(openkey) + nopen, dptr(openpos));

  thrust::lower_bound(par, dptr(openkey), dptr(openkey) + nopen,
                      dptr(g.qkey.data), dptr(g.qkey.data) + n,
                      dptr(grow(g.qidx, n)));
  launch("branch", branch_kernel, n, n, commands.data, depth, g.qidx.data,
         openkey, openpos, n1, grow(g.branch, n));
  return maxdepth;
}

// Fill g.prefix[i] = within-branch exclusive compose-scan
void resolve_prefix(const device_buffer<unsigned char> &commands,
                    const turtle_config &cfg) {
  const int n = commands.size;
  const double da = radians(cfg.angle_deg);
  auto locals = thrust::make_transform_iterator(
      dptr(commands.data), to_local{cfg.step, std::cos(da), std::sin(da)});

  int *skey = grow(g.skey, n), *order = grow(g.order, n);
  thrust::copy(par, dptr(g.branch.data), dptr(g.branch.data) + n, dptr(skey));
  thrust::sequence(par, dptr(order), dptr(order) + n);
  thrust::stable_sort_by_key(par, dptr(skey), dptr(skey) + n, dptr(order));

  affine2 *sorted = grow(g.sorted, n);
  thrust::gather(par, dptr(order), dptr(order) + n, locals, dptr(sorted));
  thrust::exclusive_scan_by_key(par, dptr(skey), dptr(skey) + n, dptr(sorted),
                                dptr(sorted), IDENTITY, thrust::equal_to<int>(),
                                compose{});
  thrust::scatter(par, dptr(sorted), dptr(sorted) + n, dptr(order),
                  dptr(grow(g.prefix, n)));
}

// Carry each '[' entry transform up the bracket tree by pointer doubling
// (parent = branch[b]), seeding each '[' with its own prefix
affine2 *resolve_entries(const device_buffer<unsigned char> &commands,
                         int maxdepth) {
  const int n = commands.size;
  affine2 *accA = grow(g.accA, n), *accB = grow(g.accB, n);
  int *jumpA = grow(g.jumpA, n), *jumpB = grow(g.jumpB, n);
  thrust::copy(par, dptr(g.prefix.data), dptr(g.prefix.data) + n, dptr(accA));
  thrust::copy(par, dptr(g.branch.data), dptr(g.branch.data) + n, dptr(jumpA));
  for (int step = 1; step < maxdepth; step <<= 1) { // chain length <= maxdepth
    launch("double", doubling_kernel, n, n, commands.data, accA, jumpA, accB,
           jumpB);
    std::swap(accA, accB);
    std::swap(jumpA, jumpB);
  }
  return accA;
}

// Same as `scan_world`
void scan_world_bracketed(const device_buffer<unsigned char> &commands,
                          const turtle_config &cfg, affine2 *world) {
  const int n = commands.size;
  if (n <= 0)
    return;
  const int maxdepth = resolve_branches(commands);
  resolve_prefix(commands, cfg);
  const affine2 *entry = resolve_entries(commands, maxdepth);

  const double h = radians(cfg.start_heading_deg);
  const affine2 init{std::cos(h), std::sin(h), 0.0, 0.0};
  launch("world_b", world_b_kernel, n, n, g.branch.data, entry, g.prefix.data,
         init, world);
  check(cudaDeviceSynchronize(), "scan_world_bracketed");
}

// Scan, then compact every drawn (`F`) frame through `xform` into `out`
template <class T, class Xform>
int scan_and_emit(const device_buffer<unsigned char> &commands,
                  const turtle_config &cfg, T *out, Xform xform) {
  const int n = commands.size;
  affine2 *world = grow(g.world, n);
  auto cmd = dptr(commands.data);
  if (thrust::count(par, cmd, cmd + n, static_cast<unsigned char>('[')) > 0)
    scan_world_bracketed(commands, cfg, world);
  else
    scan_world(commands, cfg, world);
  auto frames = thrust::make_transform_iterator(dptr(world), xform);
  auto end =
      thrust::copy_if(par, frames, frames + n, cmd, dptr(out), is_forward{});
  check(cudaDeviceSynchronize(), "sync");
  return static_cast<int>(end - dptr(out));
}

} // namespace

device_buffer<segment>
interpret_device(const device_buffer<unsigned char> &commands,
                 const turtle_config &cfg) {
  segment *out = device_alloc<segment>(commands.size); // holds at most n F's
  return {out, scan_and_emit(commands, cfg, out, to_segment{cfg.step})};
}

frames_view interpret_to_frames(const device_buffer<unsigned char> &commands,
                                const turtle_config &cfg, gpu_frame *out,
                                int out_capacity) {
  (void)out_capacity;
  int count = scan_and_emit(commands, cfg, out, to_frame{});
  return {out, count, frames_bounds(out, count, static_cast<float>(cfg.step))};
}

frames_view
interpret_frames_fallback(const device_buffer<unsigned char> &commands,
                          const turtle_config &cfg) {
  // Everything on the GPU except the D2H copy into pinned memory
  gpu_frame *dev = grow(g.frames, commands.size);
  frames_view r = interpret_to_frames(commands, cfg, dev, commands.size);
  download(g.host_ptr(r.count > 0 ? r.count : 1), dev, r.count);
  return {g.host, r.count, r.bbox};
}

std::vector<gpu_frame> interpret_frames_gpu(const std::string &commands,
                                            const turtle_config &cfg) {
  frames_view fv = interpret_frames_fallback(to_device(commands), cfg);
  return std::vector<gpu_frame>(fv.data, fv.data + fv.count);
}

std::vector<segment> to_host(const device_buffer<segment> &buf) {
  std::vector<segment> out(buf.size);
  download(out.data(), buf.data, buf.size);
  return out;
}

std::vector<segment> interpret_gpu(const std::string &commands,
                                   const turtle_config &cfg) {
  return to_host(interpret_device(to_device(commands), cfg));
}
