#include <stdexcept>

#include "datatypes.hpp"
#include "grid.hpp"

Grid::Grid(int3 size, int3 origin) {
  setSize(size);
  setOrigin(origin);
}

void Grid::setSize(int3 size) {
  if (size.x < 0 || size.y < 0 || size.z < 0) {
    throw std::invalid_argument("The grid size should be larger than or equal to 0");
  }
  size_ = size;
}

void Grid::setOrigin(int3 origin) {
  origin_ = origin;
}

bool operator==(const Grid& lhs, const Grid& rhs) {
  return lhs.size_ == rhs.size_ && lhs.origin_ == rhs.origin_;
}

bool operator!=(const Grid& lhs, const Grid& rhs) {
  return lhs.size_ != rhs.size_ || lhs.origin_ != rhs.origin_;
}
