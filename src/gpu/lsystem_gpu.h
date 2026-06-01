#pragma once

#include "core/lsystem_types.h"
#include "gpu/device_buffer.h"

#include <string>

// Upload a string to the device / download a device string back to the host
device_buffer<unsigned char> to_device(const std::string &s);
std::string to_host(const device_buffer<unsigned char> &buf);

// Expand on the GPU, leaving the result on the device
device_buffer<unsigned char> expand_device(const l_system &sys, int iterations);

// Expand on the GPU and return the result on the host
std::string expand_gpu(const l_system &sys, int iterations);
