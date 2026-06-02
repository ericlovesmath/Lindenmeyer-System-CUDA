#include "cpu/examples.h"
#include "cpu/lsystem.h"
#include "cpu/turtle.h"
#include "gpu/lsystem_gpu.h"
#include "gpu/transform_gpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr char HEADER_FMT[] = "%-11s%6s%9s%9s%9s%9s%9s\n";
constexpr char LEAD_FMT[] = "%-11s%6d%9s%9.1f";

using clk = std::chrono::steady_clock;
double ms_since(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// CPU timing
struct cpu_result {
  double ms;
  std::size_t count; // symbols or segments
};

// GPU timing, `dev` is the on-device cost
struct gpu_result {
  double dev;
  double e2e;
};

double median_of(std::vector<double> &v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

std::string human(std::size_t n) {
  char buf[16];
  if (n >= 1'000'000) {
    std::snprintf(buf, sizeof buf, "%.1fM", n / 1e6);
  } else if (n >= 1'000) {
    std::snprintf(buf, sizeof buf, "%.1fK", n / 1e3);
  } else {
    std::snprintf(buf, sizeof buf, "%zu", n);
  }
  return buf;
}

// Time `run` `reps` times (one untimed warm-up first); return the median.
template <class Run> cpu_result time_cpu(Run run, int reps) {
  run(); // warm-up: exclude first-touch allocation / page faults
  std::vector<double> samples;
  std::size_t count = 0;
  for (int i = 0; i < reps; ++i) {
    auto t0 = clk::now();
    auto result = run();
    auto t1 = clk::now();
    samples.push_back(ms_since(t0, t1));
    count = result.size();
  }
  return {median_of(samples), count};
}

// Time a GPU op `reps` times
template <class MakeDev> gpu_result time_gpu(MakeDev make_dev, int reps) {
  to_host(make_dev()); // warm-up
  std::vector<double> dev, e2e;
  for (int i = 0; i < reps; ++i) {
    auto t0 = clk::now();
    auto buf = make_dev(); // upload + compute (synchronous)
    auto t1 = clk::now();
    to_host(buf); // D2H (synchronous)
    auto t2 = clk::now();
    dev.push_back(ms_since(t0, t1));
    e2e.push_back(ms_since(t0, t2));
  }
  return {median_of(dev), median_of(e2e)};
}

void print_header(const char *count_label) {
  std::printf(HEADER_FMT, "system", "iters", count_label, "cpu ms", "gpu ms",
              "e2e ms", "speedup");
  std::printf(HEADER_FMT, "------", "-----", "-----", "------", "------",
              "------", "-------");
}

struct bench_case {
  const example &ex;
  int iterations;
};

// Time the CPU and GPU work for one case and print the row.
template <class Cpu, class MakeDev>
void bench_row(const bench_case &c, Cpu cpu_run, MakeDev make_dev, int reps) {
  cpu_result cpu = time_cpu(cpu_run, reps);
  std::printf(LEAD_FMT, c.ex.name.c_str(), c.iterations,
              human(cpu.count).c_str(), cpu.ms);
  gpu_result gpu = time_gpu(make_dev, reps);
  double speedup = gpu.dev > 0.0 ? cpu.ms / gpu.dev : 0.0;
  std::printf("%9.1f%9.1f%8.1fx\n", gpu.dev, gpu.e2e, speedup);
}

const std::vector<bench_case> expand_cases = {
    {koch, 12}, {plant, 12}, {dragon, 24}, {hilbert, 12}, {sierpinski, 16},
};

const std::vector<bench_case> transform_cases = {
    {koch, 10}, {plant, 11}, {dragon, 22}, {hilbert, 11}, {sierpinski, 13},
};

} // namespace

int main(int argc, char **argv) {
  int reps = argc > 1 ? std::atoi(argv[1]) : 5;
  if (reps < 1) {
    reps = 1;
  }

  std::printf("L-system CPU vs GPU benchmark, median of %d reps\n", reps);

  // expand
  std::printf("\n[expand] string rewriting\n");
  print_header("symbols");
  for (const bench_case &c : expand_cases) {
    bench_row(
        c, [&] { return expand(c.ex.sys, c.iterations); },
        [&] { return expand_device(c.ex.sys, c.iterations); }, reps);
  }

  // transform
  std::printf("\n[transform] turtle interpretation\n");
  print_header("segments");
  for (const bench_case &c : transform_cases) {
    std::string commands = expand(c.ex.sys, c.iterations); // untimed
    bench_row(
        c, [&] { return interpret(commands, c.ex.cfg); },
        [&] { return interpret_device(to_device(commands), c.ex.cfg); }, reps);
  }

  // pipeline
  std::printf("\n[pipeline] expand + interpret\n");
  print_header("segments");
  for (const bench_case &c : transform_cases) {
    bench_row(
        c,
        [&] {
          std::string s = expand(c.ex.sys, c.iterations);
          return interpret(s, c.ex.cfg);
        },
        [&] {
          return interpret_device(expand_device(c.ex.sys, c.iterations),
                                  c.ex.cfg);
        },
        reps);
  }
}
