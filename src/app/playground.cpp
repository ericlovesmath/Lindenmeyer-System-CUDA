// Interactive ImGui playground for the GPU L-system

#define GL_GLEXT_PROTOTYPES
#define GLFW_INCLUDE_GLEXT

#include "cpu/examples.h"
#include "cpu/image.h"
#include "cpu/turtle.h"
#include "gpu/lsystem_gpu.h"
#include "gpu/transform_gpu.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// Examples selectable from the playground UI, in display order.
const example *examples[] = {&koch, &plant, &dragon, &hilbert, &sierpinski};

void glfw_error_callback(int error, const char *desc) {
  std::fprintf(stderr, "GLFW error %d: %s\n", error, desc);
}

GLuint compile_shader(GLenum type, const char *src) {
  GLuint sh = glCreateShader(type);
  glShaderSource(sh, 1, &src, nullptr);
  glCompileShader(sh);
  GLint ok = 0;
  glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[1024];
    glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
    std::fprintf(stderr, "shader compile error: %s\n", log);
  }
  return sh;
}

// Vertex shader applies a packed affine transform, Attribute 0 is 2D position
GLuint make_line_program() {
  const char *vs = R"(#version 130
in vec2 pos;
uniform vec4 xform;
void main() { gl_Position = vec4(pos*xform.xy + xform.zw, 0.0, 1.0); })";
  const char *fs = R"(#version 130
uniform vec3 line_color;
out vec4 frag;
void main() { frag = vec4(line_color, 1.0); })";
  GLuint prog = glCreateProgram();
  glAttachShader(prog, compile_shader(GL_VERTEX_SHADER, vs));
  glAttachShader(prog, compile_shader(GL_FRAGMENT_SHADER, fs));
  glBindAttribLocation(prog, 0, "pos");
  glLinkProgram(prog);
  return prog;
}

// Draws line segments on the GPU as antialiased GL_LINES, scaled to fit
struct line_renderer {
  GLuint prog = 0, vao = 0, vbo = 0;
  GLint loc_xform = -1, loc_color = -1;
  GLsizei count = 0;
  float xform[4] = {1, 1, 0, 0}; // packed (sx, sy, tx, ty)

  void init() {
    prog = make_line_program();
    loc_xform = glGetUniformLocation(prog, "xform");
    loc_color = glGetUniformLocation(prog, "line_color");
    // Record the constant vertex layout once, attribute 0 is two floats/vertex
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindVertexArray(0);
  }

  // Upload segment endpoints and recompute the fit transform
  void set_segments(const std::vector<segment> &segs) {
    count = static_cast<GLsizei>(segs.size() * 2);
    if (segs.empty())
      return;

    double min_x = segs[0].a.x, max_x = min_x, min_y = segs[0].a.y,
           max_y = min_y;
    std::vector<float> verts;
    verts.reserve(segs.size() * 4);
    for (const segment &s : segs)
      for (const pos &p : {s.a, s.b}) {
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
        verts.push_back(static_cast<float>(p.x));
        verts.push_back(static_cast<float>(p.y));
      }

    double span = std::max({max_x - min_x, max_y - min_y, 1e-9});
    double scale = 2.0 * (1.0 - 0.05) / span; // 5% margin
    double cx = (min_x + max_x) / 2.0, cy = (min_y + max_y) / 2.0;
    xform[0] = xform[1] = static_cast<float>(scale);
    xform[2] = static_cast<float>(-cx * scale);
    xform[3] = static_cast<float>(-cy * scale);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(verts.size() * sizeof(float)),
                 verts.data(), GL_DYNAMIC_DRAW);
  }

  void draw(color c) const {
    glUseProgram(prog);
    glUniform4f(loc_xform, xform[0], xform[1], xform[2], xform[3]);
    glUniform3f(loc_color, c.r / 255.0f, c.g / 255.0f, c.b / 255.0f);
    glBindVertexArray(vao);
    glDrawArrays(GL_LINES, 0, count);
    glBindVertexArray(0);
    glUseProgram(0);
  }
};

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

// Create the window + GL context (4x MSAA) and bring up Dear ImGui
GLFWwindow *init_window() {
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit())
    return nullptr;
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_SAMPLES, 4); // MSAA window
  GLFWwindow *window =
      glfwCreateWindow(1280, 880, "L-system playground", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    return nullptr;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1); // vsync
  std::fprintf(stderr, "GL_RENDERER: %s\n", glGetString(GL_RENDERER));

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::GetIO().IniFilename = nullptr; // don't persist layout to imgui.ini
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 130");
  return window;
}

} // namespace

int main() {
  GLFWwindow *window = init_window();
  if (!window) {
    return 1;
  }

  line_renderer lines;
  lines.init();

  // Cap the iterations slider: koch grows 4x/iter, so 10 is ~3M segments
  // (~90 ms GPU / ~200 ms CPU) while 11 jumps to ~12M and stalls for ~1 s.
  constexpr int MAX_ITERS = 10;

  // The selected example and the editable parameters of the current one
  int example_idx = 0;
  const example *current = examples[example_idx];
  int iterations = std::min(current->iterations, MAX_ITERS);
  float angle = current->cfg.angle_deg;
  float heading = current->cfg.start_heading_deg;
  bool gpu_transform = true; // GPU prefix-scan turtle (non-bracketed only)

  // Stats from the last recompute. raster_ms is an exponential moving average.
  std::size_t symbols = 0, segments = 0;
  double expand_ms = 0.0, turtle_ms = 0.0, raster_ms = 0.0;

  auto recompute = [&] {
    using clock = std::chrono::steady_clock;
    using ms = std::chrono::duration<double, std::milli>;
    auto t0 = clock::now();
    std::string commands = expand_gpu(current->sys, iterations);
    auto t1 = clock::now();
    turtle_config cfg{current->cfg.step, angle, heading};
    // interpret_gpu falls back to interpret() for bracketed systems.
    auto segs =
        gpu_transform ? interpret_gpu(commands, cfg) : interpret(commands, cfg);
    auto t2 = clock::now();
    lines.set_segments(segs);
    expand_ms = ms(t1 - t0).count();
    turtle_ms = ms(t2 - t1).count();
    symbols = commands.size();
    segments = segs.size();
  };

  // Reset the editable parameters to the selected example's defaults.
  auto select_example = [&](int idx) {
    example_idx = idx;
    current = examples[idx];
    iterations = std::min(current->iterations, MAX_ITERS);
    angle = current->cfg.angle_deg;
    heading = current->cfg.start_heading_deg;
  };

  // Warm up CUDA initialization
  expand_gpu(current->sys, 1);
  recompute();

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    bool dirty = false;
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(380, 280), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls");
    ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.7f);
    if (ImGui::BeginCombo("example", current->name.c_str())) {
      for (int i = 0; i < static_cast<int>(std::size(examples)); ++i) {
        bool selected = i == example_idx;
        if (ImGui::Selectable(examples[i]->name.c_str(), selected)) {
          select_example(i);
          dirty = true;
        }
        if (selected)
          ImGui::SetItemDefaultFocus();
      }
      ImGui::EndCombo();
    }
    dirty |= ImGui::SliderInt("iterations", &iterations, 0, MAX_ITERS);
    dirty |= ImGui::SliderFloat("angle (deg)", &angle, 0.0f, 180.0f);
    dirty |= ImGui::SliderFloat("heading (deg)", &heading, 0.0f, 360.0f);
    dirty |= ImGui::Checkbox("GPU transform (non-bracketed)", &gpu_transform);
    ImGui::PopItemWidth();
    ImGui::Separator();
    ImGui::Text("%zu symbols, %zu segments", symbols, segments);
    ImGui::Text("GPU expand: %.3f ms", expand_ms);
    ImGui::Text("turtle:     %.3f ms (%s)", turtle_ms,
                gpu_transform ? "GPU" : "CPU");
    ImGui::Text("GPU raster: %.3f ms (avg)", raster_ms);
    ImGui::Text("total:      %.3f ms", expand_ms + turtle_ms + raster_ms);
    std::string ppm_path = "out/" + current->name + ".ppm";
    bool save = ImGui::Button(("Save PPM (" + ppm_path + ")").c_str());
    ImGui::End();

    if (dirty) {
      recompute();
    }

    // Centered square viewport keeps the aspect ratio
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    int s = std::min(w, h), x0 = (w - s) / 2, y0 = (h - s) / 2;

    glViewport(0, 0, w, h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f); // dark border around the canvas
    glClear(GL_COLOR_BUFFER_BIT);

    // White canvas + example, clipped to the square
    glEnable(GL_SCISSOR_TEST);
    glViewport(x0, y0, s, s);
    glScissor(x0, y0, s, s);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    auto t = std::chrono::steady_clock::now();
    lines.draw(plant.line);
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
