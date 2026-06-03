#pragma once

// Internal header shared by gpu/interpret.cu and gpu/bracket_resolve.cu

#include "core/turtle_commands.h"
#include "core/turtle_types.h"
#include "gpu/cuda_check.h"
#include "gpu/device_buffer.h"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

template <class T> inline thrust::device_ptr<T> dptr(T *p) {
  return thrust::device_pointer_cast(p);
}
inline const auto par = thrust::cuda::par_nosync;

// A rigid transform in SE(3)
struct frame3 {
  quat q;
  double px, py, pz;
};

struct compose {
  __host__ __device__ frame3 operator()(const frame3 &a,
                                        const frame3 &b) const {
    vec3 rb = qrotate(a.q, {b.px, b.py, b.pz});
    return {qmul(a.q, b.q), a.px + rb.x, a.py + rb.y, a.pz + rb.z};
  }
};

// A command symbol's local transform
struct to_local {
  double step, delta;
  __host__ __device__ frame3 operator()(unsigned char ch) const {
    turtle_action a = classify(ch, delta);
    if (a.op == turtle_op::move)
      return {{1.0, 0.0, 0.0, 0.0}, step, 0.0, 0.0};
    if (a.op == turtle_op::turn)
      return {quat_axis_angle(a.axis, a.angle), 0.0, 0.0, 0.0};
    return {{1.0, 0.0, 0.0, 0.0}, 0.0, 0.0, 0.0};
  }
};

// Grow `b` in place, only when it must hold more
template <class T> T *grow(device_buffer<T> &b, int n) {
  if (b.size < n)
    b = device_buffer<T>(device_alloc<T>(n), n);
  return b.data;
}

// All the bracket resolver's own scratch
struct bracket_scratch;

// Owns the reusable device buffers so a recompute allocates nothing.
struct turtle_engine {
  device_buffer<frame3> world;     // per-symbol world transforms
  device_buffer<gpu_frame> frames; // compacted F frames (host-copy fallback)
  gpu_frame *host = nullptr;       // pinned D2H staging
  int host_cap = 0;
  bracket_scratch *brk = nullptr; // allocated on the first bracketed system

  ~turtle_engine();
  gpu_frame *host_ptr(int n);

  // Fill `world`, then compact each drawn frame through `xform` into `out`
  template <class T, class Xform>
  int scan_and_emit(const device_buffer<unsigned char> &commands,
                    const turtle_config &cfg, T *out, Xform xform);

  // Bracketed world fill, resolves the turtle stack on-device.
  void fill_world_bracketed(const device_buffer<unsigned char> &commands,
                            const turtle_config &cfg, frame3 *world);
};
