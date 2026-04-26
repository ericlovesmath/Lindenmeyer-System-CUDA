#include "add_kernel.h"

#include <cstdio>
#include <cstdlib>

#define EXPECT_EQ(a, b)                                                        \
  do {                                                                         \
    auto _va = (a);                                                            \
    auto _vb = (b);                                                            \
    if (_va != _vb) {                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s == %s (%d vs %d)\n", __FILE__,      \
                   __LINE__, #a, #b, (int)_va, (int)_vb);                      \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

int main() {
  EXPECT_EQ(add_with_cuda(2, 3), 5);
  EXPECT_EQ(add_with_cuda(0, 0), 0);
  EXPECT_EQ(add_with_cuda(-7, 10), 3);
  EXPECT_EQ(add_with_cuda(1000, 2345), 3345);
  std::printf("All tests passed.\n");
  return 0;
}
