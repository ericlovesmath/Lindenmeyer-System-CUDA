// OpenGL implementation of frame_renderer. All shader source, the cylinder
// mesh, and the GL calls live here; the rest of the app sees only renderer.h.

#define GL_GLEXT_PROTOTYPES
#define GLFW_INCLUDE_GLEXT

#include "app/renderer.h"

#include "gpu/interpret.h" // interpret_to_frames, interpret_frames_fallback

#include <GLFW/glfw3.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr int CYL_SIDES = 10;

// GLSL helper shared by both shaders: rotate v by unit quaternion q.
const char *QROT_GLSL =
    "vec3 qrot(vec4 q, vec3 v){ return v + 2.0*cross(q.xyz, cross(q.xyz, v) + "
    "q.w*v); }\n";

GLuint compile_shader(GLenum type, const std::string &src) {
  GLuint sh = glCreateShader(type);
  const char *c = src.c_str();
  glShaderSource(sh, 1, &c, nullptr);
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

GLuint link_program(const std::string &vs, const std::string &fs) {
  GLuint prog = glCreateProgram();
  glAttachShader(prog, compile_shader(GL_VERTEX_SHADER, vs));
  glAttachShader(prog, compile_shader(GL_FRAGMENT_SHADER, fs));
  glLinkProgram(prog);
  return prog;
}

// Flat instanced-capsule shader for planar systems: project each segment to the
// xy plane and draw an antialiased screen-space capsule.
GLuint make_capsule_program() {
  std::string vs = std::string(R"(#version 330 core
layout(location = 0) in vec2 corner;
layout(location = 1) in vec3 i_pos;    // instance world position
layout(location = 2) in vec4 i_quat;   // instance orientation (x, y, z, w)
uniform mat4 u_view_proj;
uniform float u_step;
uniform float u_width_px;
uniform float u_viewport;
out vec2 v_local;
flat out float v_len_px;
)") + QROT_GLSL +
                   R"(void main() {
  vec2 tpos = i_pos.xy;
  vec2 dir = qrot(i_quat, vec3(1.0, 0.0, 0.0)).xy;
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
  return link_program(vs, fs);
}

// Instanced 3D cylinder shader. Each instance is a turtle frame; the unit tube
// is oriented along the heading (+x), scaled to the step length and radius,
// lit.
GLuint make_cylinder_program() {
  std::string vs = std::string(R"(#version 330 core
layout(location = 0) in vec3 a_mesh;   // (along in [0,1], ny, nz) on a unit tube
layout(location = 1) in vec3 i_pos;    // instance world position
layout(location = 2) in vec4 i_quat;   // instance orientation (x, y, z, w)
uniform mat4 u_view_proj;
uniform float u_step;
uniform float u_radius;
out vec3 v_normal;
)") + QROT_GLSL +
                   R"(void main() {
  vec3 local  = vec3(a_mesh.x * u_step, a_mesh.y * u_radius, a_mesh.z * u_radius);
  vec3 lnorm  = vec3(0.0, a_mesh.y, a_mesh.z);
  vec3 wpos   = i_pos + qrot(i_quat, local);
  v_normal    = qrot(i_quat, lnorm);
  gl_Position = u_view_proj * vec4(wpos, 1.0);
})";
  const char *fs = R"(#version 330 core
in vec3 v_normal;
uniform vec3 u_color;
out vec4 frag;
void main() {
  vec3 n = normalize(v_normal);
  vec3 L = normalize(vec3(0.35, 0.7, 0.55));
  float ndl = max(dot(n, L), 0.0);
  frag = vec4(u_color * (0.35 + 0.65 * ndl), 1.0);
})";
  return link_program(vs, fs);
}

// Build a unit cylinder along +x (length 1, radius 1) as (along, ny, nz) verts.
void build_cylinder(std::vector<float> &verts, std::vector<unsigned> &idx) {
  for (int i = 0; i <= CYL_SIDES; ++i) {
    float a = 2.0f * 3.14159265358979f * static_cast<float>(i) / CYL_SIDES;
    float ny = std::cos(a), nz = std::sin(a);
    verts.insert(verts.end(), {0.0f, ny, nz}); // base ring
    verts.insert(verts.end(), {1.0f, ny, nz}); // tip ring
  }
  for (int i = 0; i < CYL_SIDES; ++i) {
    unsigned b = 2u * i;
    idx.insert(idx.end(), {b, b + 1, b + 3, b, b + 3, b + 2});
  }
}

} // namespace

// Bind the shared per-instance attributes (pos at 0, quat at 12) on the
// currently bound VAO + inst_vbo.
void frame_renderer::bind_instance_attribs() {
  glBindBuffer(GL_ARRAY_BUFFER, inst_vbo_);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(gpu_frame), nullptr);
  glVertexAttribDivisor(1, 1);
  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(gpu_frame),
                        reinterpret_cast<void *>(3 * sizeof(float)));
  glVertexAttribDivisor(2, 1);
}

void frame_renderer::init() {
  prog2d_ = make_capsule_program();
  u2_ = {glGetUniformLocation(prog2d_, "u_view_proj"),
         glGetUniformLocation(prog2d_, "u_step"),
         glGetUniformLocation(prog2d_, "u_width_px"),
         glGetUniformLocation(prog2d_, "u_viewport"),
         glGetUniformLocation(prog2d_, "u_color")};

  prog3d_ = make_cylinder_program();
  u3_ = {glGetUniformLocation(prog3d_, "u_view_proj"),
         glGetUniformLocation(prog3d_, "u_step"),
         glGetUniformLocation(prog3d_, "u_radius"),
         glGetUniformLocation(prog3d_, "u_color")};

  glGenBuffers(1, &inst_vbo_);

  // 2D: a unit quad expanded into a capsule in the vertex shader.
  const float quad[] = {0.f, -0.5f, 1.f, -0.5f, 0.f, 0.5f, 1.f, 0.5f};
  glGenVertexArrays(1, &vao2d_);
  glGenBuffers(1, &quad_vbo_);
  glBindVertexArray(vao2d_);
  glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
  bind_instance_attribs();

  // 3D: an indexed cylinder mesh.
  std::vector<float> verts;
  std::vector<unsigned> idx;
  build_cylinder(verts, idx);
  index_count_ = static_cast<GLsizei>(idx.size());
  glGenVertexArrays(1, &vao3d_);
  glGenBuffers(1, &cyl_vbo_);
  glGenBuffers(1, &cyl_ebo_);
  glBindVertexArray(vao3d_);
  glBindBuffer(GL_ARRAY_BUFFER, cyl_vbo_);
  glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(),
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cyl_ebo_);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(unsigned),
               idx.data(), GL_STATIC_DRAW);
  bind_instance_attribs();
  glBindVertexArray(0);

  reserve(1 << 16);
}

// Ensure the instance buffer holds at least `frames`
void frame_renderer::reserve(int frames) {
  if (frames <= capacity_)
    return;
  int newcap = capacity_ > 0 ? capacity_ : 1;
  while (newcap < frames)
    newcap *= 2;
  reg_ = registered_buffer{}; // drop the old registration before reallocating
  glBindBuffer(GL_ARRAY_BUFFER, inst_vbo_);
  glBufferData(GL_ARRAY_BUFFER,
               static_cast<GLsizeiptr>(newcap) * sizeof(gpu_frame), nullptr,
               GL_DYNAMIC_DRAW);
  capacity_ = newcap;
  glBindVertexArray(vao2d_);
  bind_instance_attribs();
  glBindVertexArray(vao3d_);
  bind_instance_attribs();
  glBindVertexArray(0);
  interop_ = try_register_gl_buffer(reg_, inst_vbo_);
}

void frame_renderer::upload_instances(const gpu_frame *frames, GLsizei n) {
  reserve(n);
  glBindBuffer(GL_ARRAY_BUFFER, inst_vbo_);
  glBufferSubData(GL_ARRAY_BUFFER, 0,
                  static_cast<GLsizeiptr>(n) * sizeof(gpu_frame), frames);
  count_ = n;
}

void frame_renderer::fill_from_device(
    const device_buffer<unsigned char> &commands, const turtle_config &cfg) {
  step_ = static_cast<float>(cfg.step);
  if (interop_)
    reserve(commands.size); // size for up to n frames; may clear interop
  if (interop_) {
    gpu_frame *dst = map_frames(reg_);
    frames_view r = interpret_to_frames(commands, cfg, dst, capacity_);
    unmap(reg_);
    count_ = static_cast<GLsizei>(r.count);
    bbox_ = r.bbox;
    return;
  }
  frames_view fv = interpret_frames_fallback(commands, cfg);
  upload_instances(fv.data, static_cast<GLsizei>(fv.count));
  bbox_ = fv.bbox;
}

void frame_renderer::draw(const draw_params &p) const {
  auto rgb = [&](GLint loc) {
    glUniform3f(loc, p.line.r / 255.0f, p.line.g / 255.0f, p.line.b / 255.0f);
  };
  if (p.three_d) {
    // Opaque lit cylinders: depth-test, no blending.
    glEnable(GL_DEPTH_TEST);
    glUseProgram(prog3d_);
    glUniformMatrix4fv(u3_.vp, 1, GL_FALSE, p.view_proj);
    glUniform1f(u3_.step, step_);
    glUniform1f(u3_.radius, p.radius);
    rgb(u3_.color);
    glBindVertexArray(vao3d_);
    glDrawElementsInstanced(GL_TRIANGLES, index_count_, GL_UNSIGNED_INT,
                            nullptr, count_);
    glDisable(GL_DEPTH_TEST);
  } else {
    // Antialiased flat capsules: alpha blend, no depth.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(prog2d_);
    glUniformMatrix4fv(u2_.vp, 1, GL_FALSE, p.view_proj);
    glUniform1f(u2_.step, step_);
    glUniform1f(u2_.width, p.width_px);
    glUniform1f(u2_.vp_px, p.viewport_px);
    rgb(u2_.color);
    glBindVertexArray(vao2d_);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, count_);
    glDisable(GL_BLEND);
  }
  glBindVertexArray(0);
  glUseProgram(0);
}
