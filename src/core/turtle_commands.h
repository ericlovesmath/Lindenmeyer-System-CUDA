#pragma once

// Single source of truth for turtle command semantics

#include "core/quat.h"

constexpr double TURTLE_PI = 3.14159265358979323846;

QUAT_HD inline double radians(double degrees) {
  return degrees * TURTLE_PI / 180.0;
}

// Turtle local frame axes: heading H = +x, left L = +y, up U = +z.
QUAT_HD inline vec3 heading_axis() { return {1.0, 0.0, 0.0}; }
QUAT_HD inline vec3 left_axis() { return {0.0, 1.0, 0.0}; }
QUAT_HD inline vec3 up_axis() { return {0.0, 0.0, 1.0}; }

enum class turtle_op : unsigned char { ignore, move, turn, push, pop };

struct turtle_action {
  turtle_op op = turtle_op::ignore;
  bool draw = false;        // move only: 'F' draws a segment, 'f' does not
  vec3 axis{0.0, 0.0, 0.0}; // turn only: rotation axis
  double angle = 0.0;       // turn only: signed rotation angle (radians)
};

QUAT_HD inline turtle_action classify(unsigned char c, double delta) {
  switch (c) {
  case 'F':
    return {turtle_op::move, true, {}, 0.0};
  case 'f':
    return {turtle_op::move, false, {}, 0.0};
  case '+':
    return {turtle_op::turn, false, up_axis(), delta};
  case '-':
    return {turtle_op::turn, false, up_axis(), -delta};
  case '&':
    return {turtle_op::turn, false, left_axis(), delta};
  case '^':
    return {turtle_op::turn, false, left_axis(), -delta};
  case '/':
    return {turtle_op::turn, false, heading_axis(), delta};
  case '\\':
    return {turtle_op::turn, false, heading_axis(), -delta};
  case '|':
    return {turtle_op::turn, false, up_axis(), TURTLE_PI};
  case '[':
    return {turtle_op::push};
  case ']':
    return {turtle_op::pop};
  default:
    return {turtle_op::ignore};
  }
}
