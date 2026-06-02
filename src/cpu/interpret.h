#pragma once

#include "core/turtle_types.h"

#include <string>
#include <vector>

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
