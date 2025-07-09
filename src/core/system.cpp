#include "system.hpp"

#include <algorithm>
#include <stdexcept>
#include <set>

#include "datatypes.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "reduce.hpp"
#include "world.hpp"

System::System(const World* world, Grid grid, GpuBuffer<bool> geometry, GpuBuffer<unsigned int> regions)
    : grid_(grid),
      world_(world),
      geometry_(geometry),
      regions_(regions) {
  if (geometry.size() != 0 && geometry.size() != grid_.ncells()) {
    throw std::runtime_error(
        "The size of the geometry buffer does not match the size of the "
        "system.");
  }
  if (regions.size() != 0 && regions.size() != grid_.ncells()) {
    throw std::runtime_error(
        "The size of the region buffer does not match the size of the "
        "system.");
  }
  if (regions.size() != 0) {
    // Filter out unique region indices
    std::vector<unsigned int> regionsVec = regions.getData();
    std::set<unsigned int> uni(regionsVec.begin(), regionsVec.end()); // The order is of no importance
    uniqueRegions = std::vector<unsigned int>(uni.begin(), uni.end());
  }
  if (geometry.size() != 0) {
    std::vector<bool> v = geometry_.getData();
    cellsInGeo_ = grid_.ncells() - std::count(v.begin(), v.end(), false);
  }
  else { cellsInGeo_ = grid_.ncells(); }
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

real3 System::origin() const {
  return cellPosition({0, 0, 0});
}

real3 System::center() const {
  int3 size = grid().size();
  real3 corner1 = cellPosition({0, 0, 0});
  real3 corner2 = cellPosition({size.x - 1, size.y - 1, size.z - 1});
  return (corner2 + corner1) / 2;
}

const GpuBuffer<bool>& System::geometry() const {
  return geometry_;
}

const GpuBuffer<unsigned int>& System::regions() const {
  return regions_;
}

void System::checkIdxInRegions(int idx) const {
  if (!idxInRegions(GpuBuffer<unsigned int>(uniqueRegions), idx)) {
    throw std::invalid_argument("The region index " + std::to_string(idx)
                                                   + " is not defined.");
  }
}

int System::cellsInGeo() const {
  return cellsInGeo_;
}

CuSystem System::cu() const {
  return CuSystem(this);
}

CuSystem::CuSystem(const System* system)
    : grid(system->grid()),
      cellsize(system->cellsize()),
      geometry(system->geometry_.get()),
      regions(system->regions_.get()) {}
