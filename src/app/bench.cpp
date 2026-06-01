#include "cpu/examples.h"
#include "cpu/lsystem.h"
#include "gpu/lsystem_gpu.h"

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

constexpr char HEADER_FMT[] = "%-12s%-8s%-14s%-12s%-12s%-12s%-12s\n";
constexpr char ROW_FMT[] = "%-12s%-8d%-14zu%-12.3f%-12.3f%-12.2f%-12.1f\n";

struct timing {
  double median_ms;
  std::size_t symbols;
};

// Times `run` `reps` times and reports the median time and the symbol count
template <class Run> timing median_ms(Run run, int reps) {
  std::vector<double> samples;
  samples.reserve(reps);
  std::size_t symbols = 0;

  for (int i = 0; i < reps; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    auto result = run();
    auto t1 = std::chrono::steady_clock::now();

    samples.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
    symbols = result.size();
  }

  std::sort(samples.begin(), samples.end());
  return {samples[samples.size() / 2], symbols};
}

} // namespace

int main(int argc, char **argv) {
  int reps = argc > 1 ? std::atoi(argv[1]) : 5;
  if (reps < 1)
    reps = 1;

  const std::vector<bench_case> cases = {
      {koch, 12},    {plant, 12},      {dragon, 24},
      {hilbert, 12}, {sierpinski, 16}, {dragon, 24},
  };

  std::printf("expand() CPU vs GPU benchmark, median of %d reps\n", reps);
  std::printf("(gpu ms is end-to-end: H2D rule/axiom upload + kernels + D2H "
              "result)\n\n");
  std::printf(HEADER_FMT, "system", "iters", "symbols", "cpu ms", "gpu ms",
              "speedup", "gpu Msym/s");
  std::printf(HEADER_FMT, "------", "-----", "-------", "------", "------",
              "-------", "----------");

  for (const bench_case &c : cases) {
    auto cpu_run = [&] { return expand(c.ex.sys, c.iterations); };
    auto gpu_run = [&] { return expand_gpu(c.ex.sys, c.iterations); };

    gpu_run(); // untimed warm-up

    timing cpu = median_ms(cpu_run, reps);
    timing gpu = median_ms(gpu_run, reps);

    double speedup = gpu.median_ms > 0.0 ? cpu.median_ms / gpu.median_ms : 0.0;
    double gpu_msym_s =
        gpu.median_ms > 0.0 ? (gpu.symbols / 1e3) / gpu.median_ms : 0.0;

    std::printf(ROW_FMT, c.ex.name.c_str(), c.iterations, cpu.symbols,
                cpu.median_ms, gpu.median_ms, speedup, gpu_msym_s);
  }
}
