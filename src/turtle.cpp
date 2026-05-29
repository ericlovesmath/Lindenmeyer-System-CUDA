#include "turtle.h"

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

  std::vector<segment> segments;
  std::vector<state> stack;
  state turtle{{0.0, 0.0}, radians(cfg.start_heading_deg)};

  for (char command : commands) {
    switch (command) {
    case 'F':
    case 'f': {
      pos next{turtle.p.x + cfg.step * std::cos(turtle.heading),
               turtle.p.y + cfg.step * std::sin(turtle.heading)};
      if (command == 'F') {
        segments.push_back({turtle.p, next});
      }
      turtle.p = next;
      break;
    }
    case '+':
      turtle.heading += delta;
      break;
    case '-':
      turtle.heading -= delta;
      break;
    case '[':
      stack.push_back(turtle);
      break;
    case ']':
      if (!stack.empty()) {
        turtle = stack.back();
        stack.pop_back();
      }
      break;
    default:
      break;
    }
  }
  return segments;
}
