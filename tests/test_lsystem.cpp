#include "lsystem.h"
#include "turtle.h"

#include "check.h"

#include <cmath>
#include <cstdio>
#include <string>

static void test_expand() {
  l_system identity{"AB", {{'A', "AA"}}};
  EXPECT_EQ(expand(identity, 1), std::string("AAB"));

  l_system doubler{"F", {{'F', "FF"}}};
  EXPECT_EQ(expand(doubler, 5).size(), size_t(32));

  EXPECT_EQ(expand(doubler, 0), std::string("F"));

  l_system plant{"-X", {{'X', "F+[[X]-X]-F[-FX]+X"}, {'F', "FF"}}};
  EXPECT_EQ(expand(plant, 1), std::string("-F+[[X]-X]-F[-FX]+X"));
}

static void test_interpret() {
  turtle_config cfg{1.0, 90.0, 0.0};

  EXPECT_EQ(interpret("FXF", cfg).size(), size_t(2));

  auto segs = interpret("F[+F]F", cfg);
  EXPECT_EQ(segs.size(), size_t(3));

  EXPECT_TRUE(std::abs(segs[2].a.x - 1.0) < 1e-9);
  EXPECT_TRUE(std::abs(segs[2].a.y - 0.0) < 1e-9);

  EXPECT_EQ(interpret("fF", cfg).size(), size_t(1));
}

int main() {
  test_expand();
  test_interpret();
  std::printf("All tests passed.\n");
  return 0;
}
