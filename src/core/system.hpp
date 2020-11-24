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

  /** Return the grid of the system. */
  Grid grid() const;

  /** Return the cellsize of the world to which the system belongs. */
  real3 cellsize() const;

  // If a system is handed to the World, the World will update name_.
  // TODO: look for a better way to do achieve this.
  friend class World;

 private:
  World* world_;
  std::string name_ = "";
  Grid grid_;
};