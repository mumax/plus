#pragma once

#include <string>

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

  /** Return the name of the system. */
  std::string name() const;

  /** Return true if the system is registered in the world in which it lives. */
  bool registered() const;

  /** Return the grid of the system. */
  Grid grid() const;

  /** Return the cellsize of the world to which the system belongs. */
  real3 cellsize() const;

 private:
  World* world_;
  Grid grid_;
};