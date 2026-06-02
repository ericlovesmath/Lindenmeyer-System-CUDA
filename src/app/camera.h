#pragma once

// View/projection math and camera state for the playground

#include "core/turtle_types.h"

#include <algorithm>
#include <cmath>

// 2D pan/zoom camera for planar systems.
struct camera2d {
  float zoom = 1.0f, pan_x = 0.0f, pan_y = 0.0f;
};

// Orbit camera for systems that leave the plane (world up = +y).
struct camera3d {
  float yaw = 0.6f, pitch = 0.35f, zoom = 1.0f;
};

namespace camera_detail {

// out = a * b, all 4x4 column-major.
inline void mat_mul(const float a[16], const float b[16], float out[16]) {
  for (int c = 0; c < 4; ++c)
    for (int r = 0; r < 4; ++r) {
      float s = 0.0f;
      for (int k = 0; k < 4; ++k)
        s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
}

inline void perspective(float fovy, float aspect, float znear, float zfar,
                        float out[16]) {
  float f = 1.0f / std::tan(0.5f * fovy);
  for (int i = 0; i < 16; ++i)
    out[i] = 0.0f;
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (zfar + znear) / (znear - zfar);
  out[11] = -1.0f;
  out[14] = (2.0f * zfar * znear) / (znear - zfar);
}

inline void look_at(const float eye[3], const float ctr[3], float out[16]) {
  float fwd[3] = {ctr[0] - eye[0], ctr[1] - eye[1], ctr[2] - eye[2]};
  float fl = std::sqrt(fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2]);
  for (float &v : fwd)
    v /= fl;
  float up[3] = {0.0f, 1.0f, 0.0f};
  float s[3] = {fwd[1] * up[2] - fwd[2] * up[1],
                fwd[2] * up[0] - fwd[0] * up[2],
                fwd[0] * up[1] - fwd[1] * up[0]};
  float sl = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
  for (float &v : s)
    v /= sl;
  float u[3] = {s[1] * fwd[2] - s[2] * fwd[1], s[2] * fwd[0] - s[0] * fwd[2],
                s[0] * fwd[1] - s[1] * fwd[0]};
  float d_s = s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2];
  float d_u = u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2];
  float d_f = fwd[0] * eye[0] + fwd[1] * eye[1] + fwd[2] * eye[2];
  float m[16] = {s[0], u[0], -fwd[0], 0.0f, s[1], u[1], -fwd[1], 0.0f,
                 s[2], u[2], -fwd[2], 0.0f, -d_s, -d_u, d_f,     1.0f};
  for (int i = 0; i < 16; ++i)
    out[i] = m[i];
}

} // namespace camera_detail

// Column-major view-projection for the 2D ortho fit
inline void make_view_proj(const bounds3 &b, const camera2d &cam,
                           float out[16]) {
  float cx = 0.5f * (b.min_x + b.max_x), cy = 0.5f * (b.min_y + b.max_y);
  float span = std::max({b.max_x - b.min_x, b.max_y - b.min_y, 1e-9f});
  float s = 2.0f * 0.95f / span * cam.zoom;
  for (int i = 0; i < 16; ++i)
    out[i] = 0.0f;
  out[0] = s;
  out[5] = s;
  out[10] = 1.0f;
  out[15] = 1.0f;
  out[12] = -cx * s + cam.pan_x;
  out[13] = -cy * s + cam.pan_y;
}

// Column-major view-projection for the orbit camera fitted to the 3D bounds.
inline void make_view_proj(const bounds3 &b, const camera3d &cam, float aspect,
                           float out[16]) {
  using namespace camera_detail;
  float ctr[3] = {0.5f * (b.min_x + b.max_x), 0.5f * (b.min_y + b.max_y),
                  0.5f * (b.min_z + b.max_z)};
  float radius = 0.5f * std::max({b.max_x - b.min_x, b.max_y - b.min_y,
                                  b.max_z - b.min_z, 1e-6f});
  const float fovy = 0.7854f; // 45 degrees
  float dist = radius / std::tan(0.5f * fovy) * 1.4f / cam.zoom;

  float cp = std::cos(cam.pitch), sp = std::sin(cam.pitch);
  float eye[3] = {ctr[0] + dist * cp * std::sin(cam.yaw), ctr[1] + dist * sp,
                  ctr[2] + dist * cp * std::cos(cam.yaw)};

  // Clip planes track the eye distance so zooming in never clips the model.
  float proj[16], view[16];
  float znear = std::max(1e-3f, dist * 0.02f);
  float zfar = dist + 4.0f * radius + 1.0f;
  perspective(fovy, aspect, znear, zfar, proj);
  look_at(eye, ctr, view);
  mat_mul(proj, view, out);
}

// Mouse update. `dx`/`dy` are this frame's drag delta in pixels, `wheel` the
// scroll amount; both should be zeroed by the caller when the UI has the mouse.
inline camera2d pan_zoom(camera2d cam, float dx, float dy, float wheel,
                         bool dragging, int viewport_px) {
  if (viewport_px <= 0)
    return cam;
  if (wheel != 0.0f)
    cam.zoom *= std::exp(wheel * 0.1f);
  if (dragging) {
    cam.pan_x += dx * 2.0f / viewport_px;
    cam.pan_y -= dy * 2.0f / viewport_px;
  }
  return cam;
}

inline camera3d orbit(camera3d cam, float dx, float dy, float wheel,
                      bool dragging, int viewport_px) {
  if (viewport_px <= 0)
    return cam;
  if (wheel != 0.0f)
    cam.zoom =
        std::max(0.05f, std::min(50.0f, cam.zoom * std::exp(wheel * 0.15f)));
  if (dragging) {
    // Turntable: drag follows the model (drag right spins it right, drag up
    // tips the top toward you). Gain is ~half a turn across the viewport.
    float k = 3.14159265f / viewport_px;
    cam.yaw -= dx * k;
    cam.pitch -= dy * k;
    const float lim = 1.5f; // keep shy of straight up/down (gimbal)
    cam.pitch = std::max(-lim, std::min(lim, cam.pitch));
  }
  return cam;
}
