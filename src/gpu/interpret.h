#pragma once

#include "core/turtle_types.h"
#include "gpu/device_buffer.h"

#include <string>
#include <vector>

// Instance frames of every drawn with their GPU-computed bounds (bounds3 lives
// in core/turtle_types.h so the renderer/camera can use it without this header)
struct frames_view {
  const gpu_frame *data;
  int count;
  bounds3 bbox;
};

// Download device segments back to the host
std::vector<segment> to_host(const device_buffer<segment> &buf);

// Interpret a (non-bracketed) device command string on the GPU
device_buffer<segment>
interpret_device(const device_buffer<unsigned char> &commands,
                 const turtle_config &cfg);

// Interpret a device command string
frames_view interpret_to_frames(const device_buffer<unsigned char> &commands,
                                const turtle_config &cfg, gpu_frame *out);

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
