#pragma once

// Frees a device allocation
void device_free(void *p) noexcept;

// Owning handle to a device allocation of `size` elements
template <class T> struct device_buffer {
  T *data = nullptr;
  int size = 0;

  device_buffer(T *data = nullptr, int size = 0) : data(data), size(size) {}
  device_buffer(device_buffer &&o) noexcept : data(o.data), size(o.size) {
    o.data = nullptr;
  }
  device_buffer(const device_buffer &) = delete;
  ~device_buffer() { device_free(data); }
};
