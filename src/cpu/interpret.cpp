#include "cpu/interpret.h"

#include <cmath>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;

constexpr double radians(double degrees) { return degrees * PI / 180.0; }

struct state {
  pos p;
  double heading;
};

} // namespace

std::vector<segment> interpret(const std::string &commands,
                               const turtle_config &cfg) {
  const double delta = radians(cfg.angle_deg);

  state t{{0.0, 0.0}, radians(cfg.start_heading_deg)};
  std::vector<state> stack;
  std::vector<segment> segments;

  for (char command : commands) {
    switch (command) {
    case 'F':
    case 'f': {
      pos start = t.p;
      t.p.x += cfg.step * std::cos(t.heading);
      t.p.y += cfg.step * std::sin(t.heading);
      if (command == 'F') {
        segments.push_back({start, t.p});
      }
      break;
    }
    case '+':
      t.heading += delta;
      break;
    case '-':
      t.heading -= delta;
      break;
    case '[':
      stack.push_back(t);
      break;
    case ']':
      if (!stack.empty()) {
        t = stack.back();
        stack.pop_back();
      }
      break;
    default:
      // Don't draw other characters
      break;
    }
  }
  return segments;
}
