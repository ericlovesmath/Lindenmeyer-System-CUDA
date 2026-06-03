#include "cpu/examples.h"
#include "cpu/expand.h"
#include "cpu/image.h"
#include "cpu/interpret.h"

#include <iostream>
#include <string>

namespace {

// Run one L-system end to end
void render(const example &e, const std::string &path) {
  std::string commands = expand(e.sys, e.iterations);
  std::vector<segment> segments = interpret(commands, e.cfg);

  image img = make_image(1024, 1024, color{255, 255, 255});
  rasterize(segments, img, e.line);
  write_ppm(img, path);

  std::cout << path << ": " << commands.size() << " symbols, "
            << segments.size() << " segments\n";
}

} // namespace

int main(int argc, char **argv) {
  std::string out_dir = argc > 1 ? argv[1] : ".";

  for (int i = 0; i < example_count; ++i) {
    const example &e = *all_examples[i];
    render(e, out_dir + "/" + e.name + ".ppm");
  }

  return 0;
}
