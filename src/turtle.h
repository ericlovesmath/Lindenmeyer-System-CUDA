#pragma once

#include <string>
#include <vector>

struct pos {
  double x, y;
};

struct segment {
  pos a, b;
};

struct turtle_config {
  double step = 1.0;              // length of F move
  double angle_deg = 90.0;        // turn angle delta for + and -
  double start_heading_deg = 0.0; // initial heading, CCW from +x axis
};

// Walk a turtle string and emit line segments:
//   F: move forward and draw a segment
//   f: move forward without drawing
//   +: turn left by delta
//   -: turn right by delta
//   [: push turtle state
//   ]: pop turtle state
// Any other symbol is ignored.
std::vector<segment> interpret(const std::string &commands,
                               const turtle_config &cfg);
