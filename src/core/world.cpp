#include "world.hpp"

#include <stdexcept>

#include "datatypes.hpp"
#include "grid.hpp"
#include "timesolver.hpp"

World::World(real3 cellsize, Grid mastergrid, int3 pbcRepetitions)
    : cellsize_(cellsize),
      mastergrid_(mastergrid),
      pbcRepetitions_(pbcRepetitions),
      timesolver_(TimeSolver::Factory::create()) {
  if (cellsize.x <= 0 || cellsize.y <= 0 || cellsize.z <= 0) {
    throw std::invalid_argument("The cell size should be larger than 0");
  }
  checkPbcRepetitions(pbcRepetitions);
  checkPbcCompatibility(mastergrid, pbcRepetitions);
}

World::World(real3 cellsize)
    : World(cellsize, Grid(int3{0, 0, 0}), int3{0, 0, 0}) {}

World::~World() {}

real World::time() const {
  return timesolver_->time();
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

const int3 World::pbcRepetitions() const {
  return pbcRepetitions_;
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

TimeSolver& World::timesolver() const {
  return *timesolver_;
}

void World::checkPbcRepetitions(const int3 pbcRepetitions) const {
  if ((pbcRepetitions.x < 0) || (pbcRepetitions.y < 0) ||
      (pbcRepetitions.z < 0))
    throw std::invalid_argument(
        "Number of pbcRepetitions should not be negative.");
}

void World::checkPbcCompatibility(const Grid mastergrid,
                                  const int3 pbcRepetitions) const {
  if (((mastergrid.size().x == 0) ^ (pbcRepetitions.x == 0)) ||
      ((mastergrid.size().y == 0) ^ (pbcRepetitions.y == 0)) ||
      ((mastergrid.size().z == 0) ^ (pbcRepetitions.z == 0))) {
    throw std::invalid_argument(
        "0 in size of mastergrid should match 0 in "
        "pbcRepetitions.");
  }
}
