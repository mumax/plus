#pragma once

#include "datatypes.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"

class World;
class CuSystem;

class System {
 public:
  /** Construct a system with a given grid which lives in a given world. */
  System(const World* world,
         Grid grid,
         GpuBuffer<bool> geometry = GpuBuffer<bool>());

  // Systems should not be copied or moved
  System(const System&) = delete;
  System& operator=(const System&) = delete;
  System(System&&) = delete;
  System& operator=(System&&) = delete;

  /** Destroy the system. */
  virtual ~System() {}

  /** Return the world to which the system belongs. */
  const World* world() const;

  /** Return the grid of the system. */
  Grid grid() const;

  /** Return the cellsize of the world to which the system belongs. */
  real3 cellsize() const;

  /** Return the position of a cell of this system in the world. */
  real3 cellPosition(int3) const;

  /** Get the geometry of the system. */
  const GpuBuffer<bool>& geometry() const;

  /** Return a CuSystem which can be copied to the gpu and be used in cuda
   * kernels. */
  CuSystem cu() const;

 private:
  const World* world_;
  Grid grid_;
  GpuBuffer<bool> geometry_;

  friend CuSystem;
};

struct CuSystem {
  explicit CuSystem(const System*);

  const Grid grid;
  const real3 cellsize;
  bool const* geometry = nullptr;

  __device__ bool inGeometry(int3 coo) const;
};

__device__ inline bool CuSystem::inGeometry(int3 coo) const {
  if (grid.cellInGrid(coo))
    if (!geometry || geometry[grid.coord2index(coo)])
      return true;
  return false;
}
