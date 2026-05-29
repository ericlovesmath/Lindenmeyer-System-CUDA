#include "examples.h"
#include "lsystem.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

struct bench_case {
  const example &ex;
  int iterations;
};

constexpr char HEADER_FMT[] = "%-12s%-8s%-16s%-14s%-12s\n";
constexpr char ROW_FMT[] = "%-12s%-8d%-16zu%-14.3f%-12.1f\n";

// Returns the median wall-clock time in milliseconds over `reps` runs.
double median_expand_ms(const l_system &sys, int iterations, int reps) {
  std::vector<double> samples;
  samples.reserve(reps);

  for (int i = 0; i < reps; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    auto result = expand(sys, iterations);
    auto t1 = std::chrono::steady_clock::now();

    samples.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());

    // Using `result` so that it is not optimized away as dead code
    if (result.empty())
      std::puts("");
  }

  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

} // namespace

int main(int argc, char **argv) {
  int reps = argc > 1 ? std::atoi(argv[1]) : 5;
  if (reps < 1)
    reps = 1;

  const std::vector<bench_case> cases = {
      {koch, 10}, {plant, 10}, {dragon, 20}, {hilbert, 9}, {sierpinski, 9},
  };

  std::printf("expand() benchmark, median of %d reps\n\n", reps);
  std::printf(HEADER_FMT, "system", "iters", "symbols", "ms", "Msym/s");
  std::printf(HEADER_FMT, "------", "-----", "-------", "----", "------");

  for (const bench_case &c : cases) {
    std::size_t symbols = expand(c.ex.sys, c.iterations).size();
    double ms = median_expand_ms(c.ex.sys, c.iterations, reps);
    double msym_s = ms > 0.0 ? (symbols / 1e3) / ms : 0.0;

    std::printf(ROW_FMT, c.ex.name.c_str(), c.iterations, symbols, ms, msym_s);
  }
}
