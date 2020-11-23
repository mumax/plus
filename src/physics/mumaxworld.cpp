#include "mumaxworld.hpp"

#include <stdexcept>
#include <string>

#include "datatypes.hpp"
#include "ferromagnet.hpp"
#include "grid.hpp"
#include "system.hpp"

MumaxWorld::MumaxWorld(real3 cellsize, Grid mastergrid)
    : World(cellsize, mastergrid), biasMagneticField({0, 0, 0}){};

MumaxWorld::~MumaxWorld() {}

Ferromagnet* MumaxWorld::addFerromagnet(Grid grid, std::string name) {
  if (!inMastergrid(grid)) {
    throw std::out_of_range(
        "Can not add ferromagnet because the grid does not fit in the "
        "mastergrid ");
  }

  for (const auto& namedSystem : systems_) {
    System* system = namedSystem.second.get();
    Ferromagnet* fm = dynamic_cast<Ferromagnet*>(system);
    if (fm != nullptr && grid.overlaps(fm->grid())) {
      throw std::out_of_range(
          "Can not add ferromagnet because it overlaps with another "
          "ferromagnet.");
    }
  }

  static int idxUnnamed = 1;
  if (name.length() == 0) {
    name = "magnet_" + std::to_string(idxUnnamed++);
  }

  if (systems_.find(name) != systems_.end()) {
    throw std::runtime_error("A system with the name '" + name +
                             "' already exists");
  }

  // Add the magnet to the this world and get a pointer to this magnet
  addSystem(std::unique_ptr<System>(new Ferromagnet(name, grid)));
  Ferromagnet* newMagnet = static_cast<Ferromagnet*>(getSystem(name));

  // Add the magnetic field of the other magnets in this magnet, and vice versa
  for (const auto& namedSystem : systems_) {
    System* system = namedSystem.second.get();
    Ferromagnet* otherMagnet = dynamic_cast<Ferromagnet*>(system);
    if (otherMagnet != nullptr) {
      otherMagnet->addMagnetField(newMagnet, MAGNETFIELDMETHOD_AUTO);
      if (otherMagnet != newMagnet) {  // Avoid adding the field on itself twice
        newMagnet->addMagnetField(otherMagnet, MAGNETFIELDMETHOD_AUTO);
      }
    }
  }

  return newMagnet;
};

Ferromagnet* MumaxWorld::getFerromagnet(std::string name) const {
  auto namedSystem = systems_.find(name);
  if (namedSystem == systems_.end())
    return nullptr;
  System* system = namedSystem->second.get();
  Ferromagnet* magnet = dynamic_cast<Ferromagnet*>(system);
  return magnet;
}