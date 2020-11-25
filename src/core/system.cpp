#include "system.hpp"

#include "datatypes.hpp"
#include "grid.hpp"
#include "world.hpp"

System::System(World* world, Grid grid) : grid_(grid), world_(world) {}

World* System::world() const {
  return world_;
}

Grid System::grid() const {
  return grid_;
}

real3 System::cellsize() const {
  return world()->cellsize();
}