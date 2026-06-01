#pragma once

#include "core/turtle_types.h"
#include "gpu/device_buffer.h"

#include <string>
#include <vector>

// Download device segments back to the host
std::vector<segment> to_host(const device_buffer<segment> &buf);

// Interpret a (non-bracketed) device command string on the GPU
device_buffer<segment>
interpret_device(const device_buffer<unsigned char> &commands,
                 const turtle_config &cfg);

// Interpret a command string and return segments on the host
std::vector<segment> interpret_gpu(const std::string &commands,
                                   const turtle_config &cfg);
