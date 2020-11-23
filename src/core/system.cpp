#include "system.hpp"

#include <stdexcept>

#include "datatypes.hpp"
#include "grid.hpp"
#include "world.hpp"

System::System(std::string name, Grid grid) : grid_(grid), name_(name) {}

World* System::world() const {
  if (world_ == nullptr) {
    throw std::runtime_error("The system is not owned by a world.");
  }
  return world_;
}

std::string System::name() const {
  return name_;
}

Grid System::grid() const {
  return grid_;
}

real3 System::cellsize() const {
  return world()->cellsize();
}