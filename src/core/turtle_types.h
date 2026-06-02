#pragma once

struct pos {
  double x, y, z;
};

struct segment {
  pos a, b;
};

// OpenGL instance attribute: turtle world position + orientation quaternion.
// The cylinder length is the uniform turtle step, so it is not stored here.
struct gpu_frame {
  float px, py, pz;     // world position of the segment start
  float qx, qy, qz, qw; // orientation quaternion (x, y, z, w)
};

struct turtle_config {
  double step = 1.0;              // length of F move
  double angle_deg = 90.0;        // turn angle delta for the rotation commands
  double start_heading_deg = 0.0; // initial heading, CCW about +z from +x axis
};

// Axis-aligned bounding box of drawn geometry (used to fit the camera).
struct bounds3 {
  float min_x, min_y, min_z, max_x, max_y, max_z;
};
