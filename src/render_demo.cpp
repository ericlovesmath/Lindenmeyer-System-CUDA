#include "image.h"
#include "lsystem.h"
#include "turtle.h"

#include <iostream>
#include <string>

namespace {

// Run one L-system end to end
void render(const l_system &sys, int iterations, const turtle_config &cfg,
            color line, const std::string &path) {
  std::string commands = expand(sys, iterations);
  std::vector<segment> segments = interpret(commands, cfg);

  image img = make_image(1024, 1024, color{255, 255, 255});
  rasterize(segments, img, line);
  write_ppm(img, path);

  std::cout << path << ": " << commands.size() << " symbols, "
            << segments.size() << " segments\n";
}

} // namespace

int main(int argc, char **argv) {
  std::string out_dir = argc > 1 ? argv[1] : ".";

  // Koch snowflake
  l_system koch{"F++F++F", {{'F', "F-F++F-F"}}};
  render(koch, 4, turtle_config{1.0, 60.0, 0.0}, color{0, 0, 0},
         out_dir + "/koch.ppm");

  // Fractal plant, omega = -X, delta = 25.
  l_system plant{"-X", {{'X', "F+[[X]-X]-F[-FX]+X"}, {'F', "FF"}}};
  render(plant, 6, turtle_config{1.0, 25.0, 90.0}, color{30, 130, 40},
         out_dir + "/plant.ppm");

  return 0;
}
