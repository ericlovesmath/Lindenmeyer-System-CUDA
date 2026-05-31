// Interactive ImGui playground

#include "examples.h"
#include "image.h"
#include "lsystem.h"
#include "turtle.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include <cstdint>
#include <cstdio>

namespace {

constexpr int kCanvas = 800; // square render canvas, in pixels

void glfw_error_callback(int error, const char *desc) {
  std::fprintf(stderr, "GLFW error %d: %s\n", error, desc);
}

} // namespace

int main() {
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit())
    return 1;

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  GLFWwindow *window =
      glfwCreateWindow(1280, 880, "L-system playground", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1); // vsync

  // Standard Dear ImGui setup with the GLFW + OpenGL3 backends
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::GetIO().IniFilename = nullptr; // don't persist layout to imgui.ini
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 130");

  // Single texture that holds current render which is completely replaced lol
  GLuint tex = 0;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  // Editable parameters
  int iterations = plant.iterations;
  float angle = plant.cfg.angle_deg;
  float heading = plant.cfg.start_heading_deg;

  // Latest rendered image
  image img = make_image(kCanvas, kCanvas, color{255, 255, 255});
  std::size_t symbols = 0, segments = 0;

  auto render = [&] {
    std::string commands = expand(plant.sys, iterations);
    auto segs =
        interpret(commands, turtle_config{plant.cfg.step, angle, heading});
    img = make_image(kCanvas, kCanvas, color{255, 255, 255});
    rasterize(segs, img, plant.line);

    // Upload the fresh pixels into the existing texture
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, img.rgb.data());
    symbols = commands.size();
    segments = segs.size();
  };

  render();
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Controls panel
    bool dirty = false;
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360, 230), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls");
    dirty |= ImGui::SliderInt("iterations", &iterations, 0, 8);
    dirty |= ImGui::SliderFloat("angle (deg)", &angle, 0.0f, 180.0f);
    dirty |= ImGui::SliderFloat("start heading (deg)", &heading, 0.0f, 360.0f);
    ImGui::Separator();
    ImGui::Text("%zu symbols, %zu segments", symbols, segments);
    if (ImGui::Button("Save PPM (out/plant.ppm)"))
      write_ppm(img, "out/plant.ppm");
    ImGui::End();

    if (dirty)
      render();

    // Render panel
    ImGui::SetNextWindowPos(ImVec2(380, 10), ImGuiCond_FirstUseEver);
    ImGui::Begin("Render");
    ImGui::Image(static_cast<ImTextureID>(static_cast<intptr_t>(tex)),
                 ImVec2(kCanvas, kCanvas));
    ImGui::End();

    // Clear the framebuffer and draw the ImGui windows on top
    ImGui::Render();
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    glViewport(0, 0, w, h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
