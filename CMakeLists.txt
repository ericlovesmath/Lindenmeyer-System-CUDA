cmake_minimum_required(VERSION 3.24)
project(LindenmeyerSystemCUDA LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES native)

include_directories(src)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

add_library(lsystem_cpu STATIC
  src/cpu/expand.cpp src/cpu/interpret.cpp src/cpu/image.cpp src/cpu/examples.cpp)

add_library(lsystem_gpu STATIC
  src/gpu/expand.cu src/gpu/interpret.cu src/gpu/gl_interop.cu)
target_link_libraries(lsystem_gpu PRIVATE OpenGL::GL)

add_executable(render src/app/render_demo.cpp)
target_link_libraries(render PRIVATE lsystem_cpu)

add_executable(bench src/app/bench.cpp)
target_link_libraries(bench PRIVATE lsystem_cpu lsystem_gpu)

# Playground: ImGui ships as loose .cpp files compiled straight from the submodule.
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui)
if(NOT EXISTS ${IMGUI_DIR}/imgui.cpp)
  message(FATAL_ERROR "extern/imgui is empty; run: git submodule update --init")
endif()
file(GLOB IMGUI_SRC CONFIGURE_DEPENDS ${IMGUI_DIR}/*.cpp)
add_executable(playground src/app/playground.cpp ${IMGUI_SRC}
  ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
  ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(playground PRIVATE ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_link_libraries(playground PRIVATE lsystem_cpu lsystem_gpu OpenGL::GL glfw)

enable_testing()
foreach(t test_expand test_interpret)
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} PRIVATE lsystem_cpu lsystem_gpu)
  add_test(NAME ${t} COMMAND ${t})
endforeach()
