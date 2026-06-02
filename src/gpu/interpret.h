#pragma once

#include "core/turtle_types.h"
#include "gpu/device_buffer.h"

#include <string>
#include <vector>

// Axis-aligned bounding box of drawn geometry
struct bounds2 {
  float min_x, min_y, max_x, max_y;
};

// Instance frames of every drawn with their GPU-computed bounds
struct frames_view {
  const gpu_frame *data;
  int count;
  bounds2 bbox;
};

// Download device segments back to the host
std::vector<segment> to_host(const device_buffer<segment> &buf);

// Interpret a (non-bracketed) device command string on the GPU
device_buffer<segment>
interpret_device(const device_buffer<unsigned char> &commands,
                 const turtle_config &cfg);

// Interpret a (non-bracketed) device command string, writing one float
// `gpu_frame` per drawn `F` directly into `out`
frames_view interpret_to_frames(const device_buffer<unsigned char> &commands,
                                const turtle_config &cfg, gpu_frame *out,
                                int out_capacity);

// Same, but for when CUDA/GL interop is unavailable
frames_view
interpret_frames_fallback(const device_buffer<unsigned char> &commands,
                          const turtle_config &cfg);

// instance frames copied into an owning vector (for benchmarks)
std::vector<gpu_frame> interpret_frames_gpu(const std::string &commands,
                                            const turtle_config &cfg);

// Interpret a command string and return segments on the host
std::vector<segment> interpret_gpu(const std::string &commands,
                                   const turtle_config &cfg);
