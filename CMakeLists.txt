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
add_library(lsystem_cpu STATIC
  src/cpu/lsystem.cpp
  src/cpu/turtle.cpp
  src/cpu/image.cpp
  src/cpu/examples.cpp
)

# GPU L-system expansion and transform engine
find_package(OpenGL REQUIRED)
add_library(lsystem_gpu STATIC
  src/gpu/lsystem_gpu.cu
  src/gpu/transform_gpu.cu
  src/gpu/gl_interop.cu
)
target_link_libraries(lsystem_gpu PRIVATE OpenGL::GL)

add_executable(render src/app/render_demo.cpp)
target_link_libraries(render PRIVATE lsystem_cpu)

# Interactive ImGui playground
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui)
if(NOT EXISTS ${IMGUI_DIR}/imgui.cpp)
  message(FATAL_ERROR "extern/imgui is empty; run: git submodule update --init")
endif()

find_package(glfw3 REQUIRED)

# Glob the GLFW and OpenGL dependencies explicitly
file(GLOB IMGUI_SRC CONFIGURE_DEPENDS ${IMGUI_DIR}/*.cpp)
add_executable(playground
  src/app/playground.cpp
  ${IMGUI_SRC}
  ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
  ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(playground PRIVATE ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_link_libraries(playground PRIVATE lsystem_cpu lsystem_gpu OpenGL::GL glfw)

add_executable(bench src/app/bench.cpp)
target_link_libraries(bench PRIVATE lsystem_cpu lsystem_gpu)

enable_testing()
add_executable(test_expand tests/test_expand.cpp)
target_link_libraries(test_expand PRIVATE lsystem_cpu lsystem_gpu)
add_test(NAME test_expand COMMAND test_expand)

add_executable(test_interpret tests/test_interpret.cpp)
target_link_libraries(test_interpret PRIVATE lsystem_cpu lsystem_gpu)
add_test(NAME test_interpret COMMAND test_interpret)
