#include "world.hpp"

#include <memory>
#include <stdexcept>
#include <string>

#include "datatypes.hpp"
#include "system.hpp"

World::World(real3 cellsize, Grid mastergrid)
    : cellsize_(cellsize), mastergrid_(mastergrid) {
  if (cellsize.x <= 0 || cellsize.y <= 0 || cellsize.z <= 0) {
    throw std::invalid_argument("The cell size should be larger than 0");
  }
};

World::~World() {}

real3 World::cellsize() const {
  return cellsize_;
}

real World::cellVolume() const {
  return cellsize_.x * cellsize_.y * cellsize_.z;
}

Grid World::mastergrid() const {
  return mastergrid_;
}

bool World::inMastergrid(Grid grid) const {
  int3 d1 = grid.origin() - mastergrid_.origin();
  int3 d2 = d1 + grid.size() - mastergrid_.size();
  if (mastergrid_.size().x > 0 && (d1.x < 0 || d2.x > 0))
    return false;
  if (mastergrid_.size().y > 0 && (d1.y < 0 || d2.y > 0))
    return false;
  if (mastergrid_.size().z > 0 && (d1.z < 0 || d2.z > 0))
    return false;
  return true;
}

void World::registerSystem(std::shared_ptr<System> newSystem,
                           std::string name) {
  if (name.empty()) {
    throw std::runtime_error("A name is needed to register a system");
  }

  if (newSystem->world() != this) {
    throw std::runtime_error(
        "Can not register the system because the system lives in another "
        "world.");
  }

  if (!newSystem->name().empty()) {
    throw std::runtime_error("The system is already registered");
  }

  if (systems_.find(name) != systems_.end()) {
    throw std::runtime_error(
        "Another system with the name '" + name +
        "' is already registered in this world, and hence can not be added.");
  }

  systems_[name] = newSystem;
}

std::shared_ptr<System> World::registeredSystem(std::string name) const {
  auto namedSystem = systems_.find(name);
  if (namedSystem == systems_.end())
    return nullptr;
  return namedSystem->second;
}

const std::map<std::string, std::shared_ptr<System>>& World::registeredSystems()
    const {
  return systems_;
}