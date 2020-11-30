#pragma once

#include <algorithm>

#include "datatypes.hpp"

class Grid {
 public:
  __host__ Grid(int3 size, int3 origin = {0, 0, 0});

  __host__ void setSize(int3);
  __host__ void setOrigin(int3);

  __device__ __host__ int3 size() const;
  __device__ __host__ int3 origin() const;
  __device__ __host__ int ncells() const;

  __device__ __host__ int3 index2coord(int idx) const;
  __device__ __host__ int coord2index(int3 coo) const;

  __device__ __host__ bool cellInGrid(int idx) const;
  __device__ __host__ bool cellInGrid(int3 coo) const;

  /// Returns the image coordinate of 'coo' inside the grid as if the grid is
  /// repeated in every direction. This function is especially usefull when grid
  /// defines a periodic simulation box.
  __device__ __host__ int3 wrap(int3 coo) const;

  __device__ __host__ bool overlaps(Grid) const;

  __host__ friend bool operator==(const Grid& lhs, const Grid& rhs);
  __host__ friend bool operator!=(const Grid& lhs, const Grid& rhs);

 private:
  int3 size_;
  int3 origin_;
};

// -----------------------------------------------------------------------
// Implementation of inline member functions

__device__ __host__ inline int3 Grid::size() const {
  return size_;
}

__device__ __host__ inline int3 Grid::origin() const {
  return origin_;
}

__device__ __host__ inline int Grid::ncells() const {
  return size_.x * size_.y * size_.z;
}

__device__ __host__ inline int3 Grid::index2coord(int idx) const {
  return {origin_.x + idx % size_.x, origin_.y + (idx / size_.x) % size_.y,
          origin_.z + idx / (size_.x * size_.y)};
}

__device__ __host__ inline int Grid::coord2index(int3 coo) const {
  coo -= origin_;
  return coo.x + coo.y * size_.x + coo.z * size_.x * size_.y;
}

__device__ __host__ inline bool Grid::cellInGrid(int idx) const {
  return idx >= 0 && idx < ncells();
}

// true modulo operation
__device__ __host__ inline int modulo(int i, int d) {
  i %= d;
  return i >= 0 ? i : i + d;
}

__device__ __host__ inline int3 Grid::wrap(int3 idx) const {
  if (size_.x > 0)
    idx.x = origin_.x + modulo(idx.x - origin_.x, size_.x);
  if (size_.y > 0)
    idx.y = origin_.y + modulo(idx.y - origin_.y, size_.y);
  if (size_.z > 0)
    idx.z = origin_.z + modulo(idx.z - origin_.z, size_.z);
  return idx;
}

__device__ __host__ inline bool Grid::cellInGrid(int3 coo) const {
  coo -= origin_;
  return coo.x >= 0 && coo.x < size_.x && coo.y >= 0 && coo.y < size_.y &&
         coo.z >= 0 && coo.z < size_.z;
}

__device__ __host__ inline bool Grid::overlaps(Grid other) const {
// When compiling for device, use min and max from cuda api.
// When compiling for host (__CUDA_ARCH__ undefined), use min and max from std.
#ifndef __CUDA_ARCH__
  using std::max;
  using std::min;
#endif

  int x1 = max(origin_.x, other.origin_.x);
  int y1 = max(origin_.y, other.origin_.y);
  int z1 = max(origin_.z, other.origin_.z);
  int x2 = min(origin_.x + size_.x, other.origin_.x + other.size_.x);
  int y2 = min(origin_.y + size_.y, other.origin_.y + other.size_.y);
  int z2 = min(origin_.z + size_.z, other.origin_.z + other.size_.z);
  return (x2 - x1) > 0 && (y2 - y1) > 0 && (z2 - z1) > 0;
}
