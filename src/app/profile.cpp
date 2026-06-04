// Deterministic single-shot driver for Nsight Compute / Nsight Systems
// Runs exactly one expand + one interpret for one system.
//
// Usage: profile [<name> [<iters>]]   (default: plant 11, then dragon 24)

#include "cpu/examples.h"
#include "gpu/expand.h"
#include "gpu/interpret.h"

#include <nvtx3/nvToolsExt.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

const example *find_example(const std::string &name) {
  for (int i = 0; i < example_count; ++i) {
    if (all_examples[i]->name == name) {
      return all_examples[i];
    }
  }
  return nullptr;
}

// One expand + one interpret, named with a per-system NVTX range.
void run_case(const example &ex, int iterations) {
  nvtxRangePushA(ex.name.c_str());
  device_buffer<unsigned char> commands = expand_device(ex.sys, iterations);
  device_buffer<segment> segs = interpret_device(commands, ex.cfg);
  nvtxRangePop();
  std::fprintf(stderr, "%-11s iters=%-3d symbols=%-12d segments=%d\n",
               ex.name.c_str(), iterations, commands.size, segs.size);
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 1) {
    const example *ex = find_example(argv[1]);
    if (!ex) {
      std::fprintf(stderr, "unknown example '%s'\n", argv[1]);
      return 1;
    }
    int iters = argc > 2 ? std::atoi(argv[2]) : ex->iterations;
    run_case(*ex, iters);
    return 0;
  }

  // One bracketed system (sort path) and one non-bracketed (scan path)
  run_case(plant, 11);
  run_case(dragon, 24);
  return 0;
}
