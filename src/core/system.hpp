#pragma once

#include "datatypes.hpp"
#include "grid.hpp"

class World;

class System {
 public:
  /** Construct a system with a given grid which lives in a given world. */
  System(World* world, Grid grid);

  /** Destroy the system. */
  virtual ~System() {}

  /** Return the world to which the system belongs. */
  World* world() const;

  /** Return the grid of the system. */
  Grid grid() const;

  /** Return the cellsize of the world to which the system belongs. */
  real3 cellsize() const;

  /** Return the position of a cell of this system in the world. */
  real3 cellPosition(int3) const;

 private:
  World* world_;
  Grid grid_;
};
