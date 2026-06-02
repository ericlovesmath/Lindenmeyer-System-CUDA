#include "gpu/transform_gpu.h"

#include "cpu/turtle.h"      // CPU interpret() fallback for bracketed systems
#include "gpu/cuda_check.h"  // check(), device_alloc(), download()
#include "gpu/lsystem_gpu.h" // to_device() for the host-string convenience

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>

#include <cfloat>
#include <cmath>
#include <string>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;

constexpr double radians(double degrees) { return degrees * PI / 180.0; }

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
  auto in = thrust::device_pointer_cast(commands.data);
  auto locals = thrust::make_transform_iterator(in, local);
  thrust::exclusive_scan(thrust::cuda::par_nosync, locals, locals + n,
                         thrust::device_pointer_cast(world), init, compose{});
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

// Scan, then compact every drawn (`F`) frame through `xform` into `out`, and
// return the number written
template <class T, class Xform>
int scan_and_emit(const device_buffer<unsigned char> &commands,
                  const turtle_config &cfg, T *out, Xform xform) {
  const int n = commands.size;
  scan_world(commands, cfg, grow(g.world, n));
  auto in = thrust::device_pointer_cast(commands.data);
  auto src = thrust::make_transform_iterator(
      thrust::device_pointer_cast(g.world.data), xform);
  auto dst = thrust::device_pointer_cast(out);
  auto end = thrust::copy_if(src, src + n, in, dst, is_forward{});
  check(cudaDeviceSynchronize(), "sync");
  return static_cast<int>(end - dst);
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
  // Bracketed systems need a turtle stack the linear scan cannot model.
  // TODO: Work on bracketed as well.
  if (commands.find_first_of("[]") != std::string::npos) {
    return interpret(commands, cfg); // CPU fallback
  }
  return to_host(interpret_device(to_device(commands), cfg));
}
