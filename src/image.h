#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "turtle.h"

struct color {
  uint8_t r, g, b;
};

// RGB8 raster image, row-major, 3 bytes per pixel
struct image {
  int width, height;
  std::vector<uint8_t> rgb;
};

image make_image(int width, int height, color background);

// Draw segments into image, scaled and centered to fit within `margin` pixels
// of the border.
void rasterize(const std::vector<segment> &segments, image &img,
               color line_color, int margin = 16);

// Write the image as a (P6) PPM file.
void write_ppm(const image &img, const std::string &path);
