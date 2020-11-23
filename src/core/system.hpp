#pragma once

#include <string>

#include "datatypes.hpp"
#include "grid.hpp"

class World;

class System {
 public:
  /** Construct a system with a given name and grid. */
  System(std::string name, Grid grid);

  /** Destroy the system. */
  virtual ~System() {}

  /** Return the world to which the system belongs.
   *  An std::runtime_error is thrown if the system does not belong to a world.
   */
  World* world() const;

  /** Return the name of the system. */
  std::string name() const;

  /** Return the grid of the system. */
  Grid grid() const;

  /** Return the cellsize of the world to which the system belongs.
   *  An std::runtime_error is thrown if the system does not belong to a world.
   */
  real3 cellsize() const;

  // If a system is handed to the World, the World has to update world_.
  // TODO: look for a better way to do achieve this.
  friend class World;

 private:
  World* world_ = nullptr;
  std::string name_;
  Grid grid_;
};