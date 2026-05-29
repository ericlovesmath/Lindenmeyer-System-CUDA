#include "image.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace {

// Byte offset of pixel (x, y) in `rgb`
size_t index(const image &img, int x, int y) {
  return (static_cast<size_t>(y) * img.width + x) * 3;
}

// Write a color at byte offset i in the buffer.
void put(image &img, size_t i, color c) {
  img.rgb[i + 0] = c.r;
  img.rgb[i + 1] = c.g;
  img.rgb[i + 2] = c.b;
}

void set_pixel(image &img, int x, int y, color c) {
  if (x < 0 || x >= img.width || y < 0 || y >= img.height) {
    return;
  }
  put(img, index(img, x, y), c);
}

// Integer Bresenham line.
void draw_line(image &img, int x0, int y0, int x1, int y1, color c) {
  int dx = std::abs(x1 - x0);
  int dy = -std::abs(y1 - y0);
  int sx = x0 < x1 ? 1 : -1;
  int sy = y0 < y1 ? 1 : -1;
  int err = dx + dy;
  while (true) {
    set_pixel(img, x0, y0, c);
    if (x0 == x1 && y0 == y1) {
      break;
    }
    int e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x0 += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y0 += sy;
    }
  }
}

} // namespace

image make_image(int width, int height, color background) {
  image img{width, height,
            std::vector<uint8_t>(static_cast<size_t>(width) * height * 3)};
  for (size_t i = 0; i < img.rgb.size(); i += 3) {
    put(img, i, background);
  }
  return img;
}

void rasterize(const std::vector<segment> &segments, image &img,
               color line_color, int margin) {
  if (segments.empty()) {
    return;
  }

  // World-space bounding box.
  double min_x = segments[0].a.x, max_x = min_x;
  double min_y = segments[0].a.y, max_y = min_y;
  for (const segment &s : segments) {
    for (const pos &p : {s.a, s.b}) {
      min_x = std::min(min_x, p.x);
      max_x = std::max(max_x, p.x);
      min_y = std::min(min_y, p.y);
      max_y = std::max(max_y, p.y);
    }
  }

  // Uniform scale that fits the box into the margined drawing area.
  double span_x = std::max(max_x - min_x, 1e-9);
  double span_y = std::max(max_y - min_y, 1e-9);
  double avail_w = img.width - 2.0 * margin;
  double avail_h = img.height - 2.0 * margin;
  double scale = std::min(avail_w / span_x, avail_h / span_y);

  // Center the scaled drawing within the image
  double off_x = margin + (avail_w - span_x * scale) / 2.0;
  double off_y = margin + (avail_h - span_y * scale) / 2.0;

  auto to_pixel = [&](const pos &p) {
    int px = static_cast<int>(std::lround(off_x + (p.x - min_x) * scale));
    int py = static_cast<int>(
        std::lround(img.height - 1 - (off_y + (p.y - min_y) * scale)));
    return std::pair<int, int>{px, py};
  };

  for (const segment &s : segments) {
    auto [x0, y0] = to_pixel(s.a);
    auto [x1, y1] = to_pixel(s.b);
    draw_line(img, x0, y0, x1, y1, line_color);
  }
}

void write_ppm(const image &img, const std::string &path) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("cannot open file for writing: " + path);
  }
  out << "P6\n" << img.width << ' ' << img.height << "\n255\n";
  out.write(reinterpret_cast<const char *>(img.rgb.data()),
            static_cast<std::streamsize>(img.rgb.size()));
}
