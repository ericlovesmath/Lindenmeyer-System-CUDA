#include "cpu/examples.h"
#include "cpu/image.h"
#include "cpu/lsystem.h"
#include "cpu/turtle.h"

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

  render(koch, out_dir + "/koch.ppm");
  render(plant, out_dir + "/plant.ppm");
  render(dragon, out_dir + "/dragon.ppm");
  render(hilbert, out_dir + "/hilbert.ppm");
  render(sierpinski, out_dir + "/sierpinski.ppm");

  return 0;
}
