#pragma once

#include <cstdio>
#include <cstdlib>

#define EXPECT_EQ(a, b)                                                        \
  do {                                                                         \
    if (!((a) == (b))) {                                                       \
      std::fprintf(stderr, "FAIL %s:%d: %s == %s\n", __FILE__, __LINE__, #a,   \
                   #b);                                                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define EXPECT_TRUE(c)                                                         \
  do {                                                                         \
    if (!(c)) {                                                                \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)
