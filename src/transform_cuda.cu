#include "transform_cuda.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

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

struct is_forward {
  __host__ __device__ bool operator()(unsigned char ch) const {
    return ch == 'F';
  }
};

} // namespace

std::vector<segment> interpret_gpu(const std::string &commands,
                                   const turtle_config &cfg) {
  // Bracketed systems need a turtle stack the linear scan cannot model
  // TODO: Work on bracketed as well
  if (commands.find_first_of("[]") != std::string::npos) {
    return interpret(commands, cfg);
  }

  const double delta = radians(cfg.angle_deg),
               h = radians(cfg.start_heading_deg);
  const to_local local{cfg.step, std::cos(delta), std::sin(delta)};
  const affine2 init{std::cos(h), std::sin(h), 0.0, 0.0};

  // Initialization
  static auto &in = *new thrust::device_vector<unsigned char>();
  static auto &world = *new thrust::device_vector<affine2>();
  static auto &segs = *new thrust::device_vector<segment>();
  in.assign(commands.begin(), commands.end());
  const int n = static_cast<int>(in.size());
  world.resize(n);
  segs.resize(n);

  // world[i], exclusive prefix product of the per-symbol local transforms.
  auto locals = thrust::make_transform_iterator(in.begin(), local);
  thrust::exclusive_scan(thrust::cuda::par_nosync, locals, locals + n,
                         world.begin(), init, compose{});

  auto frames =
      thrust::make_transform_iterator(world.begin(), to_segment{cfg.step});
  auto end = thrust::copy_if(frames, frames + n, in.begin(), segs.begin(),
                             is_forward{});

  std::vector<segment> out(end - segs.begin());
  thrust::copy(segs.begin(), end, out.begin());
  return out;
}
