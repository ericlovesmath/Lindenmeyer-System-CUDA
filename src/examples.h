#pragma once

#include "image.h"
#include "lsystem.h"
#include "turtle.h"

#include <string>

// A named L-system together with the settings used to render it
struct example {
  std::string name;
  l_system sys;
  int iterations;     // iteration depth used by the render demo
  turtle_config cfg;  // step, turn angle, start heading
  color line;         // line color for the rasterized image
};

// Built-in examples
extern const example koch;
extern const example plant;
extern const example dragon;
extern const example hilbert;
extern const example sierpinski;
