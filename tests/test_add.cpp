#include "add_kernel.h"

#include "check.h"

// Some tests just to make sure CUDA harness works
int main() {
  EXPECT_EQ(add_with_cuda(2, 3), 5);
  EXPECT_EQ(add_with_cuda(0, 0), 0);
  EXPECT_EQ(add_with_cuda(-7, 10), 3);
  EXPECT_EQ(add_with_cuda(1000, 2345), 3345);
  std::printf("All tests passed.\n");
  return 0;
}
