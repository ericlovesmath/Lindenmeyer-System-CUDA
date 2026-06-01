#include "cpu/examples.h"
#include "cpu/lsystem.h"
#include "gpu/lsystem_gpu.h"

#include "check.h"

#include <cstdio>
#include <string>

// CPU expand() reference behavior
static void test_expand_cpu() {
  l_system identity{"AB", {{'A', "AA"}}};
  EXPECT_EQ(expand(identity, 1), std::string("AAB"));

  l_system doubler{"F", {{'F', "FF"}}};
  EXPECT_EQ(expand(doubler, 5).size(), size_t(32));

  EXPECT_EQ(expand(doubler, 0), std::string("F"));

  l_system plant{"-X", {{'X', "F+[[X]-X]-F[-FX]+X"}, {'F', "FF"}}};
  EXPECT_EQ(expand(plant, 1), std::string("-F+[[X]-X]-F[-FX]+X"));
}

static void test_expand_examples() {
  EXPECT_EQ(expand(dragon.sys, 1), std::string("FX+YF+"));
  EXPECT_EQ(expand(hilbert.sys, 1), std::string("+YF-XFX-FY+"));
}

// expand_gpu() must match expand()
static void test_gpu_matches_cpu() {
  const example *cases[] = {&koch, &plant, &dragon, &hilbert, &sierpinski};
  for (const example *e : cases) {
    for (int n = 0; n <= 6; ++n) {
      EXPECT_EQ(expand_gpu(e->sys, n), expand(e->sys, n));
    }
  }
}

// Edge cases on CPU and GPU
static void test_deletion_and_empty() {
  // rule mapping to ""
  l_system del{"ABA", {{'B', ""}}};
  EXPECT_EQ(expand(del, 1), std::string("AA"));

  l_system grow_del{"ABA", {{'A', "AB"}, {'B', ""}}};
  EXPECT_EQ(expand(grow_del, 1), std::string("ABAB"));
  EXPECT_EQ(expand(grow_del, 2), std::string("ABAB"));

  // Empty axiom stays empty
  l_system empty{"", {{'A', "AA"}}};
  EXPECT_EQ(expand(empty, 0), std::string(""));
  EXPECT_EQ(expand(empty, 5), std::string(""));

  l_system identity{"XYZ", {{'A', "BB"}}};
  EXPECT_EQ(expand(identity, 3), std::string("XYZ"));

  // GPU agrees with CPU
  const l_system *cases[] = {&del, &grow_del, &empty, &identity};
  for (const l_system *s : cases) {
    for (int n = 0; n <= 4; ++n) {
      EXPECT_EQ(expand_gpu(*s, n), expand(*s, n));
    }
  }
}

int main() {
  test_expand_cpu();
  test_expand_examples();
  test_gpu_matches_cpu();
  test_deletion_and_empty();
  std::printf("All tests passed.\n");
  return 0;
}
