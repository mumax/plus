#include "system.hpp"

#include "datatypes.hpp"
#include "grid.hpp"
#include "world.hpp"

System::System(const World* world, Grid grid) : grid_(grid), world_(world) {}

const World* System::world() const {
  return world_;
}

Grid System::grid() const {
  return grid_;
}

real3 System::cellsize() const {
  return world()->cellsize();
}

real3 System::cellPosition(int3 idx) const {
  int3 p = grid().origin() + idx;
  real3 c = cellsize();
  return real3{p.x * c.x, p.y * c.y, p.z * c.z};
}
