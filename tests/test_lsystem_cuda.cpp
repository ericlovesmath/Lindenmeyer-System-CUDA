#include "cpu/examples.h"
#include "cpu/lsystem.h"
#include "gpu/lsystem_gpu.h"

#include "check.h"

#include <cstdio>
#include <string>

static void test_matches_cpu() {
  const example *cases[] = {&koch, &plant, &dragon, &hilbert, &sierpinski};
  for (const example *e : cases) {
    for (int n = 0; n <= 6; ++n) {
      EXPECT_EQ(expand_gpu(e->sys, n), expand(e->sys, n));
    }
  }
}

static void test_expand() {
  l_system identity{"AB", {{'A', "AA"}}};
  EXPECT_EQ(expand(identity, 1), std::string("AAB"));

  l_system doubler{"F", {{'F', "FF"}}};
  EXPECT_EQ(expand(doubler, 5).size(), size_t(32));

  EXPECT_EQ(expand(doubler, 0), std::string("F"));

  l_system plant{"-X", {{'X', "F+[[X]-X]-F[-FX]+X"}, {'F', "FF"}}};
  EXPECT_EQ(expand(plant, 1), std::string("-F+[[X]-X]-F[-FX]+X"));
}

int main() {
  test_matches_cpu();
  test_expand();
  std::printf("All tests passed.\n");
  return 0;
}
