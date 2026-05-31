cmake_minimum_required(VERSION 3.24)
project(LindenmeyerSystemCUDA LANGUAGES CXX CUDA)

# Default to an optimized build
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES native)

include_directories(src)

# CPU L-system
add_library(lsystem_lib STATIC
  src/lsystem.cpp
  src/turtle.cpp
  src/image.cpp
  src/examples.cpp
)

# GPU L-system expansion
add_library(lsystem_gpu STATIC src/lsystem_cuda.cu)

add_executable(render src/render_demo.cpp)
target_link_libraries(render PRIVATE lsystem_lib)

# Interactive ImGui playground
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui)
if(NOT EXISTS ${IMGUI_DIR}/imgui.cpp)
  message(FATAL_ERROR "extern/imgui is empty; run: git submodule update --init")
endif()

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# Glob the GLFW and OpenGL dependencies explicitly
file(GLOB IMGUI_SRC CONFIGURE_DEPENDS ${IMGUI_DIR}/*.cpp)
add_executable(playground
  src/playground.cpp
  ${IMGUI_SRC}
  ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
  ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(playground PRIVATE ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_link_libraries(playground PRIVATE lsystem_lib lsystem_gpu OpenGL::GL glfw)

add_executable(bench src/bench.cpp)
target_link_libraries(bench PRIVATE lsystem_lib lsystem_gpu)

enable_testing()
add_executable(test_lsystem tests/test_lsystem.cpp)
target_link_libraries(test_lsystem PRIVATE lsystem_lib)
add_test(NAME test_lsystem COMMAND test_lsystem)

add_executable(test_lsystem_cuda tests/test_lsystem_cuda.cpp)
target_link_libraries(test_lsystem_cuda PRIVATE lsystem_lib lsystem_gpu)
add_test(NAME test_lsystem_cuda COMMAND test_lsystem_cuda)
