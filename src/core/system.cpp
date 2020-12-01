#include "system.hpp"

#include <stdexcept>

#include "datatypes.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "world.hpp"

System::System(const World* world, Grid grid, GpuBuffer<bool> geometry)
    : grid_(grid), world_(world), geometry_(geometry) {
  if (geometry.size() != 0 && geometry.size() != grid_.ncells()) {
    throw std::runtime_error(
        "The size of the geometry buffer does not match the size of the "
        "system.");
  }
}

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

const GpuBuffer<bool>& System::geometry() const {
  return geometry_;
}

CuSystem System::cu() const {
  return CuSystem(this);
}

CuSystem::CuSystem(const System* system)
    : grid(system->grid()),
      cellsize(system->cellsize()),
      geometry(system->geometry_.get()) {}
