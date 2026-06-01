#pragma once

#include "turtle.h"

#include <string>
#include <vector>

// GPU equivalent of `interpret()` for non-bracketed systems
std::vector<segment> interpret_gpu(const std::string &commands,
                                   const turtle_config &cfg);
