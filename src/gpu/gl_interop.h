#pragma once

#include "core/turtle_types.h"

#include <utility>

// CUDA side of the OpenGL interop.

struct cudaGraphicsResource; // avoids a CUDA header here

struct registered_buffer {
  cudaGraphicsResource *res = nullptr; // CUDA handle for the GL buffer
  unsigned int vbo = 0;                // the OpenGL buffer name

  registered_buffer() = default;
  registered_buffer(registered_buffer &&o) noexcept : res(o.res), vbo(o.vbo) {
    o.res = nullptr;
  }
  registered_buffer(const registered_buffer &) = delete;
  registered_buffer &operator=(registered_buffer &&o) noexcept {
    std::swap(res, o.res);
    std::swap(vbo, o.vbo);
    return *this;
  }
  ~registered_buffer();
};

// Try to register an existing GL buffer for CUDA write-discard access, with the
// GL context current. Associates CUDA with the device for the GL context first
bool try_register_gl_buffer(registered_buffer &buf, unsigned int vbo);

// Map the buffer for CUDA and return its device pointer
gpu_frame *map_frames(registered_buffer &buf);
void unmap(registered_buffer &buf);
