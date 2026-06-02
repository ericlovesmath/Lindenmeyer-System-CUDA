#pragma once

// OpenGL renderer for turtle frames, owns every GL object

#include "core/turtle_types.h" // gpu_frame, bounds3, turtle_config
#include "cpu/image.h"         // color
#include "gpu/device_buffer.h" // device_buffer
#include "gpu/gl_interop.h"    // registered_buffer

#include <GL/gl.h>

// Per-draw inputs chosen by the app.
struct draw_params {
  const float *view_proj;
  float viewport_px; // 2D: canvas pixel size (capsule antialiasing)
  color line;
  float width_px; // 2D capsule width
  float radius;   // 3D cylinder radius (world units)
  bool three_d;
};

class frame_renderer {
public:
  void init();
  // Resolve the turtle on the GPU (brackets included) into the instance buffer.
  void fill_from_device(const device_buffer<unsigned char> &commands,
                        const turtle_config &cfg);
  void draw(const draw_params &p) const;

  bool interop() const { return interop_; }
  int count() const { return count_; }
  float step() const { return step_; }
  const bounds3 &bbox() const { return bbox_; }

private:
  void reserve(int frames);
  void upload_instances(const gpu_frame *frames, GLsizei n);
  void bind_instance_attribs();

  GLuint prog2d_ = 0, prog3d_ = 0, vao2d_ = 0, vao3d_ = 0;
  GLuint quad_vbo_ = 0, cyl_vbo_ = 0, cyl_ebo_ = 0, inst_vbo_ = 0;
  struct {
    GLint vp, step, width, vp_px, color;
  } u2_{}; // capsule uniforms
  struct {
    GLint vp, step, radius, color;
  } u3_{}; // cylinder uniforms
  GLsizei index_count_ = 0;
  int capacity_ = 0;  // frames the instance buffer can hold
  GLsizei count_ = 0; // frames in the current drawing
  registered_buffer reg_;
  bool interop_ = false;           // zero-copy CUDA/GL interop available?
  bounds3 bbox_{0, 0, 0, 1, 1, 1}; // bounds of the current drawing
  float step_ = 1.0f;              // world length of one segment
};
