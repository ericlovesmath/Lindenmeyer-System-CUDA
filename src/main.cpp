#include "add_kernel.h"

#include <iostream>

int main(int argc, char **argv) {
  int a = 2, b = 3;
  if (argc == 3) {
    a = std::stoi(argv[1]);
    b = std::stoi(argv[2]);
  }
  int sum = add_with_cuda(a, b);
  std::cout << a << " + " << b << " = " << sum << std::endl;
  return 0;
}
