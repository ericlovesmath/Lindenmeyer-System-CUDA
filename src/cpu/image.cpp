#include "cpu/image.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace {

// Paint pixel (x, y), ignoring anything outside the image.
void set_pixel(image &img, int x, int y, color c) {
  if (x < 0 || x >= img.width || y < 0 || y >= img.height) {
    return;
  }
  size_t i = (static_cast<size_t>(y) * img.width + x) * 3;
  img.rgb[i + 0] = c.r;
  img.rgb[i + 1] = c.g;
  img.rgb[i + 2] = c.b;
}

// Alpha-blend `c` over pixel (x, y)
void blend_pixel(image &img, int x, int y, color c, double alpha) {
  if (x < 0 || x >= img.width || y < 0 || y >= img.height) {
    return;
  }
  alpha = std::clamp(alpha, 0.0, 1.0);

  auto mix = [alpha](uint8_t dst, uint8_t src) {
    return static_cast<uint8_t>(std::lround(dst + (src - dst) * alpha));
  };

  size_t i = (static_cast<size_t>(y) * img.width + x) * 3;
  img.rgb[i + 0] = mix(img.rgb[i + 0], c.r);
  img.rgb[i + 1] = mix(img.rgb[i + 1], c.g);
  img.rgb[i + 2] = mix(img.rgb[i + 2], c.b);
}

// Wu's Algorithm for Antialiased line
void draw_line(image &img, double x0, double y0, double x1, double y1,
               color c) {
  bool steep = std::abs(y1 - y0) > std::abs(x1 - x0);
  if (steep) {
    std::swap(x0, y0);
    std::swap(x1, y1);
  }
  if (x0 > x1) {
    std::swap(x0, x1);
    std::swap(y0, y1);
  }

  double dx = x1 - x0;
  double gradient = dx == 0.0 ? 1.0 : (y1 - y0) / dx;

  // Blend the two pixels straddling height y at column x
  auto plot = [&](int x, double y) {
    int iy = static_cast<int>(std::floor(y));
    double frac = y - iy;
    if (steep) {
      blend_pixel(img, iy, x, c, 1.0 - frac);
      blend_pixel(img, iy + 1, x, c, frac);
    } else {
      blend_pixel(img, x, iy, c, 1.0 - frac);
      blend_pixel(img, x, iy + 1, c, frac);
    }
  };

  int start = static_cast<int>(std::round(x0));
  int end = static_cast<int>(std::round(x1));
  double y = y0 + gradient * (start - x0);
  for (int x = start; x <= end; ++x) {
    plot(x, y);
    y += gradient;
  }
}

} // namespace

image make_image(int width, int height, color background) {
  image img{width, height,
            std::vector<uint8_t>(static_cast<size_t>(width) * height * 3)};
  if (background.r == background.g && background.g == background.b) {
    std::fill(img.rgb.begin(), img.rgb.end(), background.r);
  } else {
    // Write the 3-byte pattern directly, no per-pixel bounds checks
    for (size_t i = 0; i < img.rgb.size(); i += 3) {
      img.rgb[i + 0] = background.r;
      img.rgb[i + 1] = background.g;
      img.rgb[i + 2] = background.b;
    }
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
    double px = off_x + (p.x - min_x) * scale;
    double py = img.height - 1 - (off_y + (p.y - min_y) * scale);
    return std::pair<double, double>{px, py};
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
