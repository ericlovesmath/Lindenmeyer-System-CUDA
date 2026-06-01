#include "examples.h"
#include "lsystem.h"
#include "transform_cuda.h"
#include "turtle.h"

#include "check.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// Largest endpoint coordinate magnitude, used to set a relative tolerance.
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

// The GPU turtle must match the CPU turtle for every non-bracketed system
void test_matches_cpu() {
  const example *cases[] = {&koch, &dragon, &hilbert, &sierpinski};
  for (const example *e : cases) {
    for (int n = 0; n <= 6; ++n) {
      std::string commands = expand(e->sys, n);
      auto gpu = interpret_gpu(commands, e->cfg);
      auto cpu = interpret(commands, e->cfg);
      expect_segments_near(gpu, cpu);
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

// Edge cases
void test_degenerate() {
  EXPECT_EQ(interpret_gpu("", turtle_config{}).size(), std::size_t(0));
  EXPECT_EQ(interpret_gpu("++--XY", turtle_config{}).size(), std::size_t(0));
}

// Bracketed systems fall back to the CPU interpreter
void test_bracketed_fallback() {
  for (int n = 0; n <= 5; ++n) {
    std::string commands = expand(plant.sys, n);
    EXPECT_EQ(interpret_gpu(commands, plant.cfg).size(),
              interpret(commands, plant.cfg).size());
  }
}

} // namespace

int main() {
  test_matches_cpu();
  test_single_forward();
  test_degenerate();
  test_bracketed_fallback();
  std::printf("All tests passed.\n");
  return 0;
}
