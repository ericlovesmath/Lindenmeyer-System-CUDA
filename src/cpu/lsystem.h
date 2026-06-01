#pragma once

#include "core/lsystem_types.h"

// Rewrite every symbol of the string in parallel, `iterations` times, to
// produce `sigma_n` (CPU)
std::string expand(const l_system &sys, int iterations);
