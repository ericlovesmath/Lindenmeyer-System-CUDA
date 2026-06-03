// Interactive ImGui playground for the GPU L-system

#define GL_GLEXT_PROTOTYPES
#define GLFW_INCLUDE_GLEXT

#include "app/camera.h"
#include "app/renderer.h"
#include "cpu/examples.h"
#include "cpu/image.h"
#include "gpu/expand.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// Iteration cap for the slider
constexpr int MAX_ITERS = 10;

void glfw_error_callback(int error, const char *desc) {
  std::fprintf(stderr, "GLFW error %d: %s\n", error, desc);
}

// The editable parameters of the current example
struct controls {
  int iterations;
  float angle_deg;
  float heading_deg;
  float line_width_px = 1.0f; // 2D capsule width
  float radius_frac = 0.2f;   // 3D cylinder radius as a fraction of the step
};

controls defaults_for(const example &e) {
  return {std::min(e.iterations, MAX_ITERS),
          static_cast<float>(e.cfg.angle_deg),
          static_cast<float>(e.cfg.start_heading_deg), 1.0f, 0.2f};
}

// What the last recompute produced, for the stats panel
struct render_stats {
  std::size_t symbols = 0, segments = 0;
  double expand_ms = 0.0, turtle_ms = 0.0;
};

render_stats recompute(const example &ex, const controls &ctl,
                       frame_renderer &lines) {
  using clock = std::chrono::steady_clock;
  using ms = std::chrono::duration<double, std::milli>;
  render_stats st;

  auto t0 = clock::now();
  device_buffer<unsigned char> commands = expand_device(ex.sys, ctl.iterations);
  auto t1 = clock::now();

  turtle_config cfg{ex.cfg.step, ctl.angle_deg, ctl.heading_deg};
  lines.fill_from_device(commands, cfg);
  st.symbols = commands.size;
  auto t2 = clock::now();

  st.expand_ms = ms(t1 - t0).count();
  st.turtle_ms = ms(t2 - t1).count();
  st.segments = lines.count();
  return st;
}

// Read the current square example region back and save it as a PPM
void save_ppm(int x, int y, int size, const std::string &path) {
  std::vector<std::uint8_t> rgba(static_cast<size_t>(size) * size * 4);
  glReadPixels(x, y, size, size, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
  image img = make_image(size, size, color{0, 0, 0});
  for (int row = 0; row < size; ++row) {
    const std::uint8_t *src =
        &rgba[static_cast<size_t>(size - 1 - row) * size * 4];
    std::uint8_t *dst = &img.rgb[static_cast<size_t>(row) * size * 3];
    for (int i = 0; i < size; ++i) {
      dst[i * 3 + 0] = src[i * 4 + 0];
      dst[i * 3 + 1] = src[i * 4 + 1];
      dst[i * 3 + 2] = src[i * 4 + 2];
    }
  }
  write_ppm(img, path);
}

// Create the window + a 3.3 core GL context and bring up Dear ImGui
GLFWwindow *init_window() {
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit())
    return nullptr;
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_DEPTH_BITS, 24);
  GLFWwindow *window =
      glfwCreateWindow(1280, 880, "L-system playground", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    return nullptr;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  std::fprintf(stderr, "GL_RENDERER: %s\n", glGetString(GL_RENDERER));

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::GetIO().IniFilename = nullptr; // don't persist layout to imgui.ini
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");
  return window;
}

// Draw the Controls panel
bool controls_panel(int &example_idx, controls &ctl, const render_stats &stats,
                    const frame_renderer &lines, double raster_ms,
                    std::string &ppm_path, bool &save) {
  const example *current = all_examples[example_idx];
  bool dirty = false;

  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(380, 300), ImGuiCond_FirstUseEver);
  ImGui::Begin("Controls");
  ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.7f);
  if (ImGui::BeginCombo("example", current->name.c_str())) {
    for (int i = 0; i < static_cast<int>(example_count); ++i) {
      bool selected = i == example_idx;
      if (ImGui::Selectable(all_examples[i]->name.c_str(), selected)) {
        example_idx = i;
        ctl = defaults_for(*all_examples[i]);
        dirty = true;
      }
      if (selected)
        ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
  current = all_examples[example_idx];

  dirty |= ImGui::SliderInt("iterations", &ctl.iterations, 0, MAX_ITERS);
  dirty |= ImGui::SliderFloat("angle (deg)", &ctl.angle_deg, 0.0f, 180.0f);
  dirty |= ImGui::SliderFloat("heading (deg)", &ctl.heading_deg, 0.0f, 360.0f);
  if (current->three_d)
    ImGui::SliderFloat("radius (x step)", &ctl.radius_frac, 0.01f, 1.0f);
  else
    ImGui::SliderFloat("line width (px)", &ctl.line_width_px, 0.5f, 8.0f);
  ImGui::PopItemWidth();

  ImGui::Separator();
  ImGui::Text("%zu symbols, %zu segments", stats.symbols, stats.segments);
  ImGui::Text("GPU expand: %.3f ms", stats.expand_ms);
  ImGui::Text("turtle:     %.3f ms (%s)", stats.turtle_ms,
              lines.interop() ? "GPU interop" : "GPU + host copy");
  ImGui::Text("GPU raster: %.3f ms (avg)", raster_ms);
  ImGui::Text("total:      %.3f ms",
              stats.expand_ms + stats.turtle_ms + raster_ms);
  ImGui::TextDisabled(current->three_d ? "scroll = zoom, drag = orbit"
                                       : "scroll = zoom, drag = pan");
  ppm_path = "out/" + current->name + ".ppm";
  save = ImGui::Button(("Save PPM (" + ppm_path + ")").c_str());
  ImGui::End();
  return dirty;
}

} // namespace

int main() {
  GLFWwindow *window = init_window();
  if (!window)
    return 1;

  frame_renderer lines;
  lines.init();
  if (!lines.interop())
    std::fprintf(
        stderr,
        "[playground] CUDA/GL interop unavailable using host-copy upload.\n");

  int example_idx = 0;
  controls ctl = defaults_for(*all_examples[example_idx]);
  camera2d cam2;
  camera3d cam3;
  double raster_ms = 0.0; // per-frame draw time, exponential moving average

  // Warm up CUDA initialization, then build the first drawing.
  {
    device_buffer<unsigned char> warm = expand_device(all_examples[0]->sys, 1);
  }
  render_stats stats = recompute(*all_examples[example_idx], ctl, lines);

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    std::string ppm_path;
    bool save = false;
    bool dirty = controls_panel(example_idx, ctl, stats, lines, raster_ms,
                                ppm_path, save);
    const example *current = all_examples[example_idx];
    const bool d3 = current->three_d;

    if (dirty)
      stats = recompute(*current, ctl, lines);

    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    int s = std::min(w, h), x0 = (w - s) / 2, y0 = (h - s) / 2;

    // Feed mouse input to the active camera (ignoring it when the UI grabs it).
    const ImGuiIO &io = ImGui::GetIO();
    bool active = !io.WantCaptureMouse;
    float wheel = active ? io.MouseWheel : 0.0f;
    bool drag = active && io.MouseDown[0];
    float view_proj[16];
    if (d3) {
      cam3 = orbit(cam3, io.MouseDelta.x, io.MouseDelta.y, wheel, drag, s);
      make_view_proj(lines.bbox(), cam3, 1.0f, view_proj);
    } else {
      cam2 = pan_zoom(cam2, io.MouseDelta.x, io.MouseDelta.y, wheel, drag, s);
      make_view_proj(lines.bbox(), cam2, view_proj);
    }

    glViewport(0, 0, w, h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f); // dark border around the canvas
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // White canvas + example, clipped to the centered square.
    glEnable(GL_SCISSOR_TEST);
    glViewport(x0, y0, s, s);
    glScissor(x0, y0, s, s);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    auto t = std::chrono::steady_clock::now();
    lines.draw({view_proj, static_cast<float>(s), current->line,
                ctl.line_width_px, ctl.radius_frac * lines.step(), d3});
    glFinish();
    double sample = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t)
                        .count();
    raster_ms = raster_ms == 0.0 ? sample : 0.9 * raster_ms + 0.1 * sample;
    if (save)
      save_ppm(x0, y0, s, ppm_path);
    glDisable(GL_SCISSOR_TEST);

    // Dear ImGui draws on top of the canvas.
    ImGui::Render();
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
