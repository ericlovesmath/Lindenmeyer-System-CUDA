#include "cpu/interpret.h"

#include "core/turtle_commands.h"

#include <vector>

namespace {

struct state {
  pos p;
  quat q; // orientation; heading direction = qrotate(q, heading_axis())
};

} // namespace

std::vector<segment> interpret(const std::string &commands,
                               const turtle_config &cfg) {
  const double delta = radians(cfg.angle_deg);

  // Initial heading is a rotation about up (+z), keeping planar systems planar.
  state t{{0.0, 0.0, 0.0},
          quat_axis_angle(up_axis(), radians(cfg.start_heading_deg))};
  std::vector<state> stack;
  std::vector<segment> segments;

  for (char command : commands) {
    const turtle_action a =
        classify(static_cast<unsigned char>(command), delta);
    switch (a.op) {
    case turtle_op::move: {
      pos start = t.p;
      vec3 h = qrotate(t.q, heading_axis());
      t.p = {start.x + cfg.step * h.x, start.y + cfg.step * h.y,
             start.z + cfg.step * h.z};
      if (a.draw)
        segments.push_back({start, t.p});
      break;
    }
    case turtle_op::turn:
      // Intrinsic rotation about a turtle-local axis, so post-multiply.
      t.q = qmul(t.q, quat_axis_angle(a.axis, a.angle));
      break;
    case turtle_op::push:
      stack.push_back(t);
      break;
    case turtle_op::pop:
      if (!stack.empty()) {
        t = stack.back();
        stack.pop_back();
      }
      break;
    case turtle_op::ignore:
      break;
    }
  }
  return segments;
}
