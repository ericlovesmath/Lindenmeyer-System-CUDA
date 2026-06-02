#include "cpu/interpret.h"

#include "core/quat.h"

#include <cmath>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;

constexpr double radians(double degrees) { return degrees * PI / 180.0; }

// Turtle local frame: heading H = +x, left L = +y, up U = +z.
constexpr vec3 HEADING{1.0, 0.0, 0.0};
constexpr vec3 LEFT{0.0, 1.0, 0.0};
constexpr vec3 UP{0.0, 0.0, 1.0};

struct state {
  pos p;
  quat q; // orientation; heading direction = qrotate(q, HEADING)
};

} // namespace

std::vector<segment> interpret(const std::string &commands,
                               const turtle_config &cfg) {
  const double delta = radians(cfg.angle_deg);

  // Initial heading is a rotation about up (+z), keeping planar systems planar.
  state t{{0.0, 0.0, 0.0},
          quat_axis_angle(UP, radians(cfg.start_heading_deg))};
  std::vector<state> stack;
  std::vector<segment> segments;

  // Rotate the frame about a turtle-local axis (intrinsic, so post-multiply).
  auto turn = [&](vec3 axis, double a) {
    t.q = qmul(t.q, quat_axis_angle(axis, a));
  };

  for (char command : commands) {
    switch (command) {
    case 'F':
    case 'f': {
      pos start = t.p;
      vec3 h = qrotate(t.q, HEADING);
      t.p = {start.x + cfg.step * h.x, start.y + cfg.step * h.y,
             start.z + cfg.step * h.z};
      if (command == 'F')
        segments.push_back({start, t.p});
      break;
    }
    case '+': turn(UP, delta); break;
    case '-': turn(UP, -delta); break;
    case '&': turn(LEFT, delta); break;
    case '^': turn(LEFT, -delta); break;
    case '/': turn(HEADING, delta); break;
    case '\\': turn(HEADING, -delta); break;
    case '|': turn(UP, PI); break;
    case '[': stack.push_back(t); break;
    case ']':
      if (!stack.empty()) {
        t = stack.back();
        stack.pop_back();
      }
      break;
    default: break;
    }
  }
  return segments;
}
