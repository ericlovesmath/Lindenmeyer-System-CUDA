#pragma once

#include "core/lsystem_types.h"

#include <string>

// GPU equivalent of `expand()`
std::string expand_gpu(const l_system &sys, int iterations);
