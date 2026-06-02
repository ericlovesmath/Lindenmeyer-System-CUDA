#include "gpu/gl_interop.h"

#include "gpu/cuda_check.h" // check()

#include <cuda_gl_interop.h>

registered_buffer::~registered_buffer() {
  if (res) {
    cudaGraphicsUnregisterResource(res);
  }
}

bool try_register_gl_buffer(registered_buffer &buf, unsigned int vbo) {
  buf = registered_buffer{}; // release any prior registration first

  unsigned int gl_count = 0;
  int gl_device = 0;
  if (cudaGLGetDevices(&gl_count, &gl_device, 1, cudaGLDeviceListAll) !=
          cudaSuccess ||
      gl_count == 0) {
    cudaGetLastError();
    return false;
  }
  if (cudaSetDevice(gl_device) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }

  cudaGraphicsResource *res = nullptr;
  if (cudaGraphicsGLRegisterBuffer(
          &res, vbo, cudaGraphicsRegisterFlagsWriteDiscard) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  buf.res = res;
  buf.vbo = vbo;
  return true;
}

gpu_frame *map_frames(registered_buffer &buf) {
  check(cudaGraphicsMapResources(1, &buf.res), "cudaGraphicsMapResources");
  gpu_frame *ptr = nullptr;
  std::size_t mapped = 0;
  check(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&ptr),
                                             &mapped, buf.res),
        "cudaGraphicsResourceGetMappedPointer");
  return ptr;
}

void unmap(registered_buffer &buf) {
  check(cudaGraphicsUnmapResources(1, &buf.res), "cudaGraphicsUnmapResources");
}
