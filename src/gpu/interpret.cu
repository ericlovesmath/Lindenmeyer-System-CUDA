#include "gpu/interpret.h"

#include "gpu/turtle_engine.h"

#include "gpu/expand.h"
#include "gpu/nvtx.h"

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>

#include <cfloat>
#include <string>
#include <vector>

namespace {

// The segment a forward step draws from a turtle frame.
struct to_segment {
  double step;
  __host__ __device__ segment operator()(const frame3 &w) const {
    vec3 h = qrotate(w.q, heading_axis());
    return {{w.px, w.py, w.pz},
            {w.px + step * h.x, w.py + step * h.y, w.pz + step * h.z}};
  }
};

// A turtle frame narrowed to a float OpenGL instance attribute.
struct to_frame {
  __host__ __device__ gpu_frame operator()(const frame3 &w) const {
    quat q = quat_normalize(w.q);
    return {static_cast<float>(w.px), static_cast<float>(w.py),
            static_cast<float>(w.pz), static_cast<float>(q.x),
            static_cast<float>(q.y),  static_cast<float>(q.z),
            static_cast<float>(q.w)};
  }
};

struct is_forward {
  __host__ __device__ bool operator()(unsigned char ch) const {
    return ch == 'F';
  }
};

gpu_frame *pinned_alloc(int count) {
  gpu_frame *p = nullptr;
  check(cudaHostAlloc(&p, sizeof(gpu_frame) * (count > 0 ? count : 1),
                      cudaHostAllocDefault),
        "cudaHostAlloc");
  return p;
}
void pinned_free(gpu_frame *p) noexcept {
  if (p)
    cudaFreeHost(p);
}

// The bounding box of the segment one frame draws.
struct frame_extent {
  float step;
  __host__ __device__ bounds3 operator()(const gpu_frame &f) const {
    quat q{f.qw, f.qx, f.qy, f.qz};
    vec3 h = qrotate(q, heading_axis());
    float ex = f.px + step * static_cast<float>(h.x);
    float ey = f.py + step * static_cast<float>(h.y);
    float ez = f.pz + step * static_cast<float>(h.z);
    return {fminf(f.px, ex), fminf(f.py, ey), fminf(f.pz, ez),
            fmaxf(f.px, ex), fmaxf(f.py, ey), fmaxf(f.pz, ez)};
  }
};
struct bounds_union {
  __host__ __device__ bounds3 operator()(const bounds3 &a,
                                         const bounds3 &b) const {
    return {fminf(a.min_x, b.min_x), fminf(a.min_y, b.min_y),
            fminf(a.min_z, b.min_z), fmaxf(a.max_x, b.max_x),
            fmaxf(a.max_y, b.max_y), fmaxf(a.max_z, b.max_z)};
  }
};
bounds3 frames_bounds(const gpu_frame *frames, int count, float step) {
  if (count == 0)
    return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const bounds3 init{FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
  auto p = thrust::device_pointer_cast(frames);
  return thrust::transform_reduce(thrust::cuda::par, p, p + count,
                                  frame_extent{step}, init, bounds_union{});
}

// Flat world fill: a matrix prefix scan with the associative `compose`.
void scan_world(const device_buffer<unsigned char> &commands,
                const turtle_config &cfg, frame3 *world) {
  NVTX_RANGE("world-scan");
  const int n = commands.size;
  const frame3 init{quat_axis_angle(up_axis(), radians(cfg.start_heading_deg)),
                    0.0, 0.0, 0.0};
  auto locals = thrust::make_transform_iterator(
      dptr(commands.data), to_local{cfg.step, radians(cfg.angle_deg)});
  thrust::exclusive_scan(par, locals, locals + n, dptr(world), init, compose{});
}

// Leaked on purpose: its buffers free through the allocator pool (gpu/pool.cu),
// so running the destructor at exit would race the pool's static teardown.
turtle_engine &engine() {
  static turtle_engine *e = new turtle_engine();
  return *e;
}

} // namespace

gpu_frame *turtle_engine::host_ptr(int n) {
  if (host_cap < n) {
    pinned_free(host);
    host = pinned_alloc(n);
    host_cap = n;
  }
  return host;
}

template <class T, class Xform>
int turtle_engine::scan_and_emit(const device_buffer<unsigned char> &commands,
                                 const turtle_config &cfg, T *out,
                                 Xform xform) {
  const int n = commands.size;
  frame3 *w = grow(world, n);
  auto cmd = dptr(commands.data);
  if (thrust::count(par, cmd, cmd + n, static_cast<unsigned char>('[')) > 0)
    fill_world_bracketed(commands, cfg, w);
  else
    scan_world(commands, cfg, w);
  auto frames = thrust::make_transform_iterator(dptr(w), xform);
  int count;
  {
    NVTX_RANGE("emit");
    auto end =
        thrust::copy_if(par, frames, frames + n, cmd, dptr(out), is_forward{});
    count = static_cast<int>(end - dptr(out));
  }
  check(cudaDeviceSynchronize(), "sync");
  return count;
}

device_buffer<segment>
interpret_device(const device_buffer<unsigned char> &commands,
                 const turtle_config &cfg) {
  segment *out = device_alloc<segment>(commands.size); // holds at most n F's
  return {out,
          engine().scan_and_emit(commands, cfg, out, to_segment{cfg.step})};
}

frames_view interpret_to_frames(const device_buffer<unsigned char> &commands,
                                const turtle_config &cfg, gpu_frame *out) {
  // `out` must hold at least commands.size frames (one per drawn 'F', at most
  // n).
  int count = engine().scan_and_emit(commands, cfg, out, to_frame{});
  return {out, count, frames_bounds(out, count, static_cast<float>(cfg.step))};
}

frames_view
interpret_frames_fallback(const device_buffer<unsigned char> &commands,
                          const turtle_config &cfg) {
  gpu_frame *dev = grow(engine().frames, commands.size);
  frames_view r = interpret_to_frames(commands, cfg, dev);
  download(engine().host_ptr(r.count > 0 ? r.count : 1), dev, r.count);
  return {engine().host, r.count, r.bbox};
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
