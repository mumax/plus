#include "system.hpp"

#include <stdexcept>

#include "datatypes.hpp"
#include "grid.hpp"
#include "world.hpp"

System::System(World* world, Grid grid) : grid_(grid), world_(world) {}

World* System::world() const {
  return world_;
}

std::string System::name() const {
  for (auto it : world()->registeredSystems()) {
    if (it.second.get() == this) {
      return it.first;
    }
  }
  return "";
}

bool System::registered() const {
  for (auto it : world()->registeredSystems()) {
    if (it.second.get() == this) {
      return true;
    }
  }
  return false;
}

Grid System::grid() const {
  return grid_;
}

real3 System::cellsize() const {
  return world()->cellsize();
}