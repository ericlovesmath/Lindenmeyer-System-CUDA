#pragma once

#include "lsystem.h"

#include <string>

// GPU equivalent of `expand()`
std::string expand_gpu(const l_system &sys, int iterations);
