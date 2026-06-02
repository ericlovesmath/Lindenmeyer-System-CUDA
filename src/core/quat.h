#pragma once

// Minimal unit-quaternion math

#ifdef __CUDACC__
#define QUAT_HD __host__ __device__
#else
#define QUAT_HD
#endif

#include <cmath>

struct vec3 {
  double x, y, z;
};

struct quat {
  double w, x, y, z;
};

QUAT_HD inline quat qmul(const quat &a, const quat &b) {
  return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
          a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
          a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
          a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}

// Rotate v by q (q assumed unit): v + 2w(u x v) + 2 u x (u x v), u = q.xyz
QUAT_HD inline vec3 qrotate(const quat &q, const vec3 &v) {
  const double ux = q.x, uy = q.y, uz = q.z;
  // t = 2 * (u x v)
  const double tx = 2.0 * (uy * v.z - uz * v.y);
  const double ty = 2.0 * (uz * v.x - ux * v.z);
  const double tz = 2.0 * (ux * v.y - uy * v.x);
  return {v.x + q.w * tx + (uy * tz - uz * ty),
          v.y + q.w * ty + (uz * tx - ux * tz),
          v.z + q.w * tz + (ux * ty - uy * tx)};
}

// Unit quaternion for a rotation of `angle` radians about a unit `axis`.
QUAT_HD inline quat quat_axis_angle(const vec3 &axis, double angle) {
  const double h = 0.5 * angle, s = std::sin(h);
  return {std::cos(h), axis.x * s, axis.y * s, axis.z * s};
}

QUAT_HD inline quat quat_normalize(const quat &q) {
  double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (n <= 0.0)
    return {1.0, 0.0, 0.0, 0.0};
  double inv = 1.0 / n;
  return {q.w * inv, q.x * inv, q.y * inv, q.z * inv};
}
