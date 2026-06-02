#include "cpu/examples.h"
#include "cpu/lsystem.h"
#include "cpu/turtle.h"
#include "gpu/transform_gpu.h"

#include "check.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// Largest endpoint coordinate magnitude
double extent(const std::vector<segment> &segs) {
  double m = 1.0;
  for (const segment &s : segs)
    for (const pos &p : {s.a, s.b})
      m = std::max({m, std::fabs(p.x), std::fabs(p.y)});
  return m;
}

void expect_segments_near(const std::vector<segment> &got,
                          const std::vector<segment> &want) {
  EXPECT_EQ(got.size(), want.size());

  double tol = 1e-9 * extent(want);
  for (std::size_t i = 0; i < want.size(); ++i) {
    EXPECT_NEAR(got[i].a.x, want[i].a.x, tol);
    EXPECT_NEAR(got[i].a.y, want[i].a.y, tol);
    EXPECT_NEAR(got[i].b.x, want[i].b.x, tol);
    EXPECT_NEAR(got[i].b.y, want[i].b.y, tol);
  }
}

// Direction vector of a segment (b - a)
pos dir(const segment &s) { return {s.b.x - s.a.x, s.b.y - s.a.y}; }

void test_interpret_cpu() {
  turtle_config cfg{1.0, 90.0, 0.0};

  EXPECT_EQ(interpret("FXF", cfg).size(), std::size_t(2));

  auto segs = interpret("F[+F]F", cfg);
  EXPECT_EQ(segs.size(), std::size_t(3));

  EXPECT_NEAR(segs[2].a.x, 1.0, 1e-9);
  EXPECT_NEAR(segs[2].a.y, 0.0, 1e-9);

  EXPECT_EQ(interpret("fF", cfg).size(), std::size_t(1));
}

void test_rotation_direction() {
  turtle_config cfg{1.0, 90.0, 0.0};

  auto plus = interpret("F+F", cfg);
  EXPECT_EQ(plus.size(), std::size_t(2));
  EXPECT_NEAR(dir(plus[0]).x, 1.0, 1e-9);
  EXPECT_NEAR(dir(plus[0]).y, 0.0, 1e-9);
  EXPECT_NEAR(dir(plus[1]).x, 0.0, 1e-9);
  EXPECT_NEAR(dir(plus[1]).y, 1.0, 1e-9);

  auto minus = interpret("F-F", cfg);
  EXPECT_NEAR(dir(minus[1]).x, 0.0, 1e-9);
  EXPECT_NEAR(dir(minus[1]).y, -1.0, 1e-9);

  expect_segments_near(interpret_gpu("F+F", cfg), plus);
  expect_segments_near(interpret_gpu("F-F", cfg), minus);
}

// ']' restores the full turtle state
void test_bracket_restores_heading() {
  turtle_config cfg{1.0, 90.0, 0.0};

  // railing F resumes the original +x heading
  auto segs = interpret("F[+++F]F", cfg);
  EXPECT_EQ(segs.size(), std::size_t(3));
  EXPECT_NEAR(segs[2].a.x, 1.0, 1e-9);
  EXPECT_NEAR(segs[2].a.y, 0.0, 1e-9);
  EXPECT_NEAR(dir(segs[2]).x, 1.0, 1e-9);
  EXPECT_NEAR(dir(segs[2]).y, 0.0, 1e-9);

  // Nested brackets restore at each ]
  auto nested = interpret("F[+F[+F]]F", cfg);
  EXPECT_EQ(nested.size(), std::size_t(4));
  EXPECT_NEAR(dir(nested[1]).y, 1.0, 1e-9);
  EXPECT_NEAR(dir(nested[2]).x, -1.0, 1e-9);
  EXPECT_NEAR(dir(nested[3]).x, 1.0, 1e-9);
  EXPECT_NEAR(nested[3].a.x, 1.0, 1e-9);
  EXPECT_NEAR(nested[3].a.y, 0.0, 1e-9);
}

// The GPU turtle must match the CPU turtle
void test_gpu_matches_cpu() {
  const example *cases[] = {&koch, &plant, &dragon, &hilbert, &sierpinski};
  for (const example *e : cases) {
    for (int n = 0; n <= 6; ++n) {
      std::string commands = expand(e->sys, n);
      expect_segments_near(interpret_gpu(commands, e->cfg),
                           interpret(commands, e->cfg));
    }
  }
}

// interpret_to_frames  must agree with the segment path
void test_frames_match_segments() {
  const example *cases[] = {&koch, &dragon, &hilbert, &sierpinski};
  for (const example *e : cases) {
    for (int n = 0; n <= 6; ++n) {
      std::string commands = expand(e->sys, n);
      auto segs = interpret(commands, e->cfg);
      auto frames = interpret_frames_gpu(commands, e->cfg);
      EXPECT_EQ(frames.size(), segs.size());
      float tol = static_cast<float>(1e-4 * extent(segs));
      double step = e->cfg.step;
      for (std::size_t i = 0; i < segs.size(); ++i) {
        EXPECT_NEAR(frames[i].tx, segs[i].a.x, tol);
        EXPECT_NEAR(frames[i].ty, segs[i].a.y, tol);
        EXPECT_NEAR(frames[i].tx + step * frames[i].c, segs[i].b.x, tol);
        EXPECT_NEAR(frames[i].ty + step * frames[i].s, segs[i].b.y, tol);
      }
    }
  }
}

void test_single_forward() {
  turtle_config cfg{2.0, 90.0, 30.0};
  auto segs = interpret_gpu("F", cfg);
  EXPECT_EQ(segs.size(), std::size_t(1));
  double h = 30.0 * 3.14159265358979323846 / 180.0;
  EXPECT_NEAR(segs[0].a.x, 0.0, 1e-12);
  EXPECT_NEAR(segs[0].a.y, 0.0, 1e-12);
  EXPECT_NEAR(segs[0].b.x, 2.0 * std::cos(h), 1e-12);
  EXPECT_NEAR(segs[0].b.y, 2.0 * std::sin(h), 1e-12);
}

void test_degenerate() {
  EXPECT_EQ(interpret_gpu("", turtle_config{}).size(), std::size_t(0));
  EXPECT_EQ(interpret_gpu("++--XY", turtle_config{}).size(), std::size_t(0));
}

void test_bracket_edge_cases() {
  turtle_config cfg{1.0, 25.0, 70.0};
  const char *cases[] = {
      "F[+F",           // unclosed '[' runs to end of string
      "F[+F[-F",        // unclosed nested branches
      "F[f+F]F",        // 'f' moves without drawing inside a branch
      "F[+F][-F]F",     // sibling branches restore independently
      "F[+F[+F[+F]]]F", // deep nesting
      "[[[F]]]",        // immediate nesting then draw
  };
  for (const char *s : cases) {
    expect_segments_near(interpret_gpu(s, cfg), interpret(s, cfg));
  }
}

// Every built-in example runs end to end
void test_examples_sanity() {
  const example *cases[] = {&koch, &plant, &dragon, &hilbert, &sierpinski};
  for (const example *e : cases) {
    std::string commands = expand(e->sys, e->iterations);
    EXPECT_TRUE(!interpret(commands, e->cfg).empty());
  }
}

} // namespace

int main() {
  test_interpret_cpu();
  test_rotation_direction();
  test_bracket_restores_heading();
  test_gpu_matches_cpu();
  test_frames_match_segments();
  test_single_forward();
  test_degenerate();
  test_bracket_edge_cases();
  test_examples_sanity();
  std::printf("All tests passed.\n");
  return 0;
}
