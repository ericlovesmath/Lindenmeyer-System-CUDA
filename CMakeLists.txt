cmake_minimum_required(VERSION 3.24)
project(LindenmeyerSystemCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES native)

include_directories(src)

# CPU L-system baseline: grammar rewriter, turtle interpreter, PPM rasterizer.
add_library(lsystem_lib STATIC
  src/lsystem.cpp
  src/turtle.cpp
  src/image.cpp
  src/examples.cpp
)

add_executable(render src/render_demo.cpp)
target_link_libraries(render PRIVATE lsystem_lib)

add_executable(bench src/bench.cpp)
target_link_libraries(bench PRIVATE lsystem_lib)

add_executable(lsystem src/main.cpp src/add_kernel.cu)

enable_testing()
add_executable(test_lsystem tests/test_lsystem.cpp)
target_link_libraries(test_lsystem PRIVATE lsystem_lib)
add_test(NAME test_lsystem COMMAND test_lsystem)

add_executable(test_add tests/test_add.cpp src/add_kernel.cu)
add_test(NAME test_add COMMAND test_add)
