cmake_minimum_required(VERSION 3.24)
project(LindenmeyerSystemCUDA LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES native)

include_directories(src)

add_executable(lsystem src/main.cpp src/add_kernel.cu)

enable_testing()
add_executable(test_add tests/test_add.cpp src/add_kernel.cu)
add_test(NAME test_add COMMAND test_add)
