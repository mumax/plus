#include "world.hpp"

#include <handler.hpp>
#include <stdexcept>
#include <string>

#include "ferromagnet.hpp"

World::World(real3 cellsize, Grid mastergrid)
    : cellsize_(cellsize),
      mastergrid_(mastergrid),
      biasMagneticField({0, 0, 0}) {
  if (cellsize.x <= 0 || cellsize.y <= 0 || cellsize.z <= 0) {
    throw std::invalid_argument("The cell size should be larger than 0");
  }
};

World::~World() {
  for (auto fm : Ferromagnets) {
    delete fm.second;
  }
}

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

Ferromagnet* World::addFerromagnet(Grid grid, std::string name) {
  for (const auto& fm : Ferromagnets)
    if (grid.overlaps(fm.second->grid()))
      throw std::out_of_range(
          "Can not add ferromagnet because it overlaps with another "
          "ferromagnet.");

  if (!inMastergrid(grid))
    throw std::out_of_range(
        "Can not add ferromagnet because the grid does not fit in the "
        "mastergrid ");

  static int idxUnnamed = 1;
  if (name.length() == 0)
    name = "magnet_" + std::to_string(idxUnnamed++);

  if (Ferromagnets.find(name) != Ferromagnets.end())
    throw std::runtime_error("A ferromagnet with the name '" + name +
                             "' already exists");
  printf("World.cpp line 67.\n");
  Ferromagnet* newMagnet = new Ferromagnet(this, name, grid);
  Ferromagnets[name] = newMagnet;
  printf("World.cpp line 70.\n");

  // Add the magnetic field of the other magnets in this magnet, and vice versa
  for (auto& entry : Ferromagnets) {
    printf("World.cpp line 74.\n");
    Ferromagnet* magnet = entry.second;
    printf("World.cpp line 76.\n");
    magnet->addMagnetField(newMagnet, MAGNETFIELDMETHOD_AUTO);
    printf("World.cpp line 78.\n");
    if (magnet != newMagnet) {  // Avoid adding the field on itself twice
      printf("World.cpp line 80.\n");
      newMagnet->addMagnetField(magnet, MAGNETFIELDMETHOD_AUTO);
      printf("World.cpp line 82.\n");
    }
  }
  printf("World.cpp line 80.");

  return newMagnet;
};

Ferromagnet* World::getFerromagnet(std::string name) const {
  auto result = Ferromagnets.find(name);
  if (result == Ferromagnets.end())
    return nullptr;
  return result->second;
}