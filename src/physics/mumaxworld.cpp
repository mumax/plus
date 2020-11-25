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

  for (const auto& namedMagnet : ferromagnets_) {
    Ferromagnet* fm = namedMagnet.second.get();
    if (grid.overlaps(fm->grid())) {
      throw std::out_of_range(
          "Can not add ferromagnet because it overlaps with another "
          "ferromagnet.");
    }
  }

  static int idxUnnamed = 1;
  if (name.length() == 0) {
    name = "magnet_" + std::to_string(idxUnnamed++);
  }

  if (ferromagnets_.find(name) != ferromagnets_.end()) {
    throw std::runtime_error("A ferromagnet with the name '" + name +
                             "' already exists");
  }

  // Create the magnet and add it to this world
  ferromagnets_[name] = std::make_unique<Ferromagnet>(this, grid, name);
  Ferromagnet* newMagnet = ferromagnets_[name].get();

  // Add the magnetic field of the other magnets in this magnet, and vice versa
  for (const auto& namedMagnet : ferromagnets_) {
    Ferromagnet* otherMagnet = namedMagnet.second.get();
    if (otherMagnet != nullptr) {
      otherMagnet->addMagnetField(newMagnet, MAGNETFIELDMETHOD_AUTO);
      // Avoid adding the field on itself twice
      if (otherMagnet != newMagnet) {
        newMagnet->addMagnetField(otherMagnet, MAGNETFIELDMETHOD_AUTO);
      }
    }
  }

  return newMagnet;
};

Ferromagnet* MumaxWorld::getFerromagnet(std::string name) const {
  auto namedMagnet = ferromagnets_.find(name);
  if (namedMagnet == ferromagnets_.end())
    return nullptr;
  return namedMagnet->second.get();
}