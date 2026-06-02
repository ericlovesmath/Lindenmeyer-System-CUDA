// Interactive ImGui playground for the GPU L-system

#define GL_GLEXT_PROTOTYPES
#define GLFW_INCLUDE_GLEXT

#include "cpu/examples.h"
#include "cpu/image.h"
#include "cpu/turtle.h"
#include "gpu/gl_interop.h"
#include "gpu/lsystem_gpu.h"
#include "gpu/transform_gpu.h"

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

// Examples selectable from the playground UI, in display order.
const example *examples[] = {&koch, &plant, &dragon, &hilbert, &sierpinski};

// Iteration cap for the slider
constexpr int MAX_ITERS = 10;

bool is_bracketed(const l_system &sys) {
  if (sys.axiom.find('[') != std::string::npos)
    return true;
  for (const auto &rule : sys.rules)
    if (rule.second.find('[') != std::string::npos)
      return true;
  return false;
}

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

// Instanced capsule shader
GLuint make_capsule_program() {
  const char *vs = R"(#version 330 core
layout(location = 0) in vec2 corner;
layout(location = 1) in vec4 i_frame;  // (tx, ty, cos, sin)
uniform mat4 u_view_proj;
uniform float u_step;
uniform float u_width_px;
uniform float u_viewport;
out vec2 v_local;          // (along_px, perp_px) relative to the core
flat out float v_len_px;   // projected core length in pixels
void main() {
  vec2 tpos = i_frame.xy;
  vec2 dir = i_frame.zw;
  vec4 a = u_view_proj * vec4(tpos, 0.0, 1.0);
  vec4 b = u_view_proj * vec4(tpos + dir * u_step, 0.0, 1.0);
  float halfvp = u_viewport * 0.5;
  v_len_px = distance(a.xy, b.xy) * halfvp;

  vec4 clip = mix(a, b, corner.x);
  vec2 perp = vec2(-dir.y, dir.x);
  float halfw = u_width_px * 0.5;
  float capdir = corner.x * 2.0 - 1.0;
  vec2 off_px = perp * (corner.y * u_width_px) + dir * (capdir * halfw);
  clip.xy += off_px / halfvp;

  v_local = vec2(corner.x * v_len_px + capdir * halfw, corner.y * u_width_px);
  gl_Position = clip;
})";
  const char *fs = R"(#version 330 core
in vec2 v_local;
flat in float v_len_px;
uniform float u_width_px;
uniform vec3 u_color;
out vec4 frag;
void main() {
  float halfw = u_width_px * 0.5;
  float x = clamp(v_local.x, 0.0, v_len_px);
  float d = length(vec2(v_local.x - x, v_local.y)) - halfw;
  float alpha = clamp(0.5 - d / fwidth(d), 0.0, 1.0);
  if (alpha <= 0.0)
    discard;
  frag = vec4(u_color, alpha);
})";
  GLuint prog = glCreateProgram();
  glAttachShader(prog, compile_shader(GL_VERTEX_SHADER, vs));
  glAttachShader(prog, compile_shader(GL_FRAGMENT_SHADER, fs));
  glLinkProgram(prog);
  return prog;
}

// Convert CPU turtle segments (the bracketed path) to instance frames
std::vector<gpu_frame> segments_to_frames(const std::vector<segment> &segs) {
  std::vector<gpu_frame> frames;
  frames.reserve(segs.size());
  for (const segment &s : segs) {
    float dx = float(s.b.x - s.a.x), dy = float(s.b.y - s.a.y);
    float len = std::sqrt(dx * dx + dy * dy);
    float inv = len > 0.0f ? 1.0f / len : 0.0f;
    frames.push_back({float(s.a.x), float(s.a.y), dx * inv, dy * inv});
  }
  return frames;
}

// Bounding box of the segments `frames` would draw for the camera fit
bounds2 host_bounds(const std::vector<gpu_frame> &frames, double step) {
  if (frames.empty())
    return {0, 0, 0, 0};
  float st = static_cast<float>(step);
  bounds2 b{frames[0].tx, frames[0].ty, frames[0].tx, frames[0].ty};
  for (const gpu_frame &f : frames) {
    float x1 = f.tx + st * f.c, y1 = f.ty + st * f.s;
    b.min_x = std::min({b.min_x, f.tx, x1});
    b.min_y = std::min({b.min_y, f.ty, y1});
    b.max_x = std::max({b.max_x, f.tx, x1});
    b.max_y = std::max({b.max_y, f.ty, y1});
  }
  return b;
}

// Per-frame draw inputs, the camera and line style
struct draw_params {
  const float *view_proj;
  float viewport_px;
  color line;
  float width_px;
};

// Draws instanced capsules from a buffer of turtle frames
struct instanced_renderer {
  GLuint prog = 0, vao = 0, quad_vbo = 0, inst_vbo = 0;
  GLint loc_vp = -1, loc_step = -1, loc_width = -1, loc_vp_px = -1,
        loc_color = -1;
  int capacity = 0;  // frames the instance buffer can hold
  GLsizei count = 0; // frames in the current drawing
  registered_buffer reg;
  bool interop = false;     // zero-copy CUDA/GL interop available?
  bounds2 bbox{0, 0, 1, 1}; // bounds of the current drawing (camera fit)
  float step = 1.0f;        // world length of one segment

  void init() {
    prog = make_capsule_program();
    loc_vp = glGetUniformLocation(prog, "u_view_proj");
    loc_step = glGetUniformLocation(prog, "u_step");
    loc_width = glGetUniformLocation(prog, "u_width_px");
    loc_vp_px = glGetUniformLocation(prog, "u_viewport");
    loc_color = glGetUniformLocation(prog, "u_color");

    const float quad[] = {0.f, -0.5f, 1.f, -0.5f, 0.f, 0.5f, 1.f, 0.5f};
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &quad_vbo);
    glGenBuffers(1, &inst_vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, quad_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, inst_vbo);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(gpu_frame), nullptr);
    glVertexAttribDivisor(1, 1);
    glBindVertexArray(0);

    reserve(1 << 16);
  }

  // Ensure the instance buffer holds at least `frames`
  void reserve(int frames) {
    if (frames <= capacity)
      return;
    int newcap = capacity > 0 ? capacity : 1;
    while (newcap < frames)
      newcap *= 2;
    reg = registered_buffer{}; // drop the old registration before reallocating
    glBindBuffer(GL_ARRAY_BUFFER, inst_vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(newcap) * sizeof(gpu_frame), nullptr,
                 GL_DYNAMIC_DRAW);
    capacity = newcap;
    interop = try_register_gl_buffer(reg, inst_vbo);
  }

  // Upload `n` frames from host memory into the instance buffer.
  void upload_instances(const gpu_frame *frames, GLsizei n) {
    reserve(n);
    glBindBuffer(GL_ARRAY_BUFFER, inst_vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0,
                    static_cast<GLsizeiptr>(n) * sizeof(gpu_frame), frames);
    count = n;
  }

  // Non-bracketed path
  void fill_from_device(const device_buffer<unsigned char> &commands,
                        const turtle_config &cfg) {
    step = static_cast<float>(cfg.step);
    if (interop)
      reserve(commands.size); // size for up to n frames; may clear interop
    if (interop) {
      gpu_frame *dst = map_frames(reg);
      frames_view r = interpret_to_frames(commands, cfg, dst, capacity);
      unmap(reg);
      count = static_cast<GLsizei>(r.count);
      bbox = r.bbox;
      return;
    }
    frames_view fv = interpret_frames_fallback(commands, cfg);
    upload_instances(fv.data, static_cast<GLsizei>(fv.count));
    bbox = fv.bbox;
  }

  // Bracketed path: frames resolved on the CPU, uploaded into the same buffer.
  void fill_from_host(const std::vector<gpu_frame> &frames, double seg_len) {
    step = static_cast<float>(seg_len);
    bbox = host_bounds(frames, seg_len);
    upload_instances(frames.data(), static_cast<GLsizei>(frames.size()));
  }

  void draw(const draw_params &p) const {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(prog);
    glUniformMatrix4fv(loc_vp, 1, GL_FALSE, p.view_proj);
    glUniform1f(loc_step, step);
    glUniform1f(loc_width, p.width_px);
    glUniform1f(loc_vp_px, p.viewport_px);
    glUniform3f(loc_color, p.line.r / 255.0f, p.line.g / 255.0f,
                p.line.b / 255.0f);
    glBindVertexArray(vao);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, count);
    glBindVertexArray(0);
    glUseProgram(0);
    glDisable(GL_BLEND);
  }
};

// User view on top of the autofit
struct camera {
  float zoom = 1.0f, pan_x = 0.0f, pan_y = 0.0f;
};

// Column-major 2D ortho
void make_view_proj(const bounds2 &b, const camera &cam, float out[16]) {
  float cx = 0.5f * (b.min_x + b.max_x), cy = 0.5f * (b.min_y + b.max_y);
  float span = std::max({b.max_x - b.min_x, b.max_y - b.min_y, 1e-9f});
  float s = 2.0f * 0.95f / span * cam.zoom;
  for (int i = 0; i < 16; ++i)
    out[i] = 0.0f;
  out[0] = s;
  out[5] = s;
  out[10] = 1.0f;
  out[15] = 1.0f;
  out[12] = -cx * s + cam.pan_x;
  out[13] = -cy * s + cam.pan_y;
}

// Apply mouse input to the camera
camera apply_mouse(camera cam, const ImGuiIO &io, int viewport_px) {
  if (io.WantCaptureMouse || viewport_px <= 0)
    return cam;
  if (io.MouseWheel != 0.0f)
    cam.zoom *= std::exp(io.MouseWheel * 0.1f);
  if (io.MouseDown[0]) {
    cam.pan_x += io.MouseDelta.x * 2.0f / viewport_px;
    cam.pan_y -= io.MouseDelta.y * 2.0f / viewport_px;
  }
  return cam;
}

// The editable parameters of the current example
struct controls {
  int iterations;
  float angle_deg;
  float heading_deg;
  float line_width_px = 1.5f;
};

controls defaults_for(const example &e) {
  return {std::min(e.iterations, MAX_ITERS),
          static_cast<float>(e.cfg.angle_deg),
          static_cast<float>(e.cfg.start_heading_deg), 1.5f};
}

// What the last recompute produced, for the stats panel
struct render_stats {
  std::size_t symbols = 0, segments = 0;
  double expand_ms = 0.0, turtle_ms = 0.0;
  bool turtle_on_gpu = true;
};

render_stats recompute(const example &ex, const controls &ctl,
                       instanced_renderer &lines) {
  using clock = std::chrono::steady_clock;
  using ms = std::chrono::duration<double, std::milli>;
  render_stats st;
  st.turtle_on_gpu = !is_bracketed(ex.sys);

  auto t0 = clock::now();
  device_buffer<unsigned char> commands = expand_device(ex.sys, ctl.iterations);
  auto t1 = clock::now();

  turtle_config cfg{ex.cfg.step, ctl.angle_deg, ctl.heading_deg};
  if (st.turtle_on_gpu) {
    lines.fill_from_device(commands, cfg);
    st.symbols = commands.size;
  } else {
    std::string s = to_host(commands);
    lines.fill_from_host(segments_to_frames(interpret(s, cfg)), cfg.step);
    st.symbols = s.size();
  }
  auto t2 = clock::now();

  st.expand_ms = ms(t1 - t0).count();
  st.turtle_ms = ms(t2 - t1).count();
  st.segments = lines.count;
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
  ImGui_ImplOpenGL3_Init("#version 330");
  return window;
}

} // namespace

int main() {
  GLFWwindow *window = init_window();
  if (!window) {
    return 1;
  }

  instanced_renderer lines;
  lines.init();
  if (!lines.interop)
    std::fprintf(
        stderr,
        "[playground] CUDA/GL interop unavailable using host-copy upload.\n");

  int example_idx = 0;
  controls ctl = defaults_for(*examples[example_idx]);
  camera cam;
  double raster_ms = 0.0; // per-frame draw time, exponential moving average

  // Warm up CUDA initialization, then build the first drawing.
  {
    device_buffer<unsigned char> warm = expand_device(examples[0]->sys, 1);
  }
  render_stats stats = recompute(*examples[example_idx], ctl, lines);

  while (!glfwWindowShouldClose(window)) {
    const example *current = examples[example_idx];
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    bool dirty = false;
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(380, 300), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls");
    ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.7f);
    if (ImGui::BeginCombo("example", current->name.c_str())) {
      for (int i = 0; i < static_cast<int>(std::size(examples)); ++i) {
        bool selected = i == example_idx;
        if (ImGui::Selectable(examples[i]->name.c_str(), selected)) {
          example_idx = i;
          ctl = defaults_for(*examples[i]); // reset params + camera to defaults
          cam = camera{};
          dirty = true;
        }
        if (selected)
          ImGui::SetItemDefaultFocus();
      }
      ImGui::EndCombo();
    }
    dirty |= ImGui::SliderInt("iterations", &ctl.iterations, 0, MAX_ITERS);
    dirty |= ImGui::SliderFloat("angle (deg)", &ctl.angle_deg, 0.0f, 180.0f);
    dirty |=
        ImGui::SliderFloat("heading (deg)", &ctl.heading_deg, 0.0f, 360.0f);
    ImGui::SliderFloat("line width (px)", &ctl.line_width_px, 0.5f, 8.0f);
    ImGui::PopItemWidth();
    ImGui::Separator();
    ImGui::Text("%zu symbols, %zu segments", stats.symbols, stats.segments);
    ImGui::Text("GPU expand: %.3f ms", stats.expand_ms);
    const char *turtle_where = !stats.turtle_on_gpu ? "CPU brackets"
                               : lines.interop      ? "GPU interop"
                                                    : "GPU + host copy";
    ImGui::Text("turtle:     %.3f ms (%s)", stats.turtle_ms, turtle_where);
    ImGui::Text("GPU raster: %.3f ms (avg)", raster_ms);
    ImGui::Text("total:      %.3f ms",
                stats.expand_ms + stats.turtle_ms + raster_ms);
    ImGui::TextDisabled("scroll = zoom, drag = pan");
    std::string ppm_path = "out/" + current->name + ".ppm";
    bool save = ImGui::Button(("Save PPM (" + ppm_path + ")").c_str());
    ImGui::End();

    if (dirty)
      stats = recompute(*current, ctl, lines);

    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    int s = std::min(w, h), x0 = (w - s) / 2, y0 = (h - s) / 2;
    cam = apply_mouse(cam, ImGui::GetIO(), s);

    glViewport(0, 0, w, h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f); // dark border around the canvas
    glClear(GL_COLOR_BUFFER_BIT);

    // White canvas + example, clipped to the centered square
    glEnable(GL_SCISSOR_TEST);
    glViewport(x0, y0, s, s);
    glScissor(x0, y0, s, s);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    float view_proj[16];
    make_view_proj(lines.bbox, cam, view_proj);

    auto t = std::chrono::steady_clock::now();
    lines.draw(
        {view_proj, static_cast<float>(s), plant.line, ctl.line_width_px});
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
