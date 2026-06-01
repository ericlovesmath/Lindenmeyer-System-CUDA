#pragma once

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
