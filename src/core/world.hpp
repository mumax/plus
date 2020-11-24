#pragma once

#include <map>
#include <memory>
#include <string>

#include "datatypes.hpp"
#include "grid.hpp"
#include "system.hpp"

class World {
 public:
  /** Create a world with a given cell size and master grid
   *  If the mastergrid has a zero size, then the mastergrid is considered to be
   *  infinitely large.
   */
  World(real3 cellsize, Grid mastergrid = Grid(int3{0, 0, 0}));

  /** Destroy the world, including all the systems it contains. */
  ~World();

  /** Return the cell size dimensions. */
  real3 cellsize() const;

  /** Return the cell volume. */
  real cellVolume() const;

  /** Return the master grid of the world.
   *  If the mastergrid has a zero size, then the mastergrid is considered to be
   *  infenitely large.
   */
  Grid mastergrid() const;

  /** Returns true if the grid is completely inside the mastergrid. */
  bool inMastergrid(Grid) const;

  /** Register a system to this world.
   *  The world becomes aware of this system, and will hold a shared pointer to
   *  the system.
   */
  void registerSystem(std::shared_ptr<System>, std::string name);

  /** Get a pointer to a system in this world by its name. */
  std::shared_ptr<System> registeredSystem(std::string name) const;

  /** Get map of all registered systems in this world. */
  const std::map<std::string, std::shared_ptr<System>>& registeredSystems()
      const;

 protected:
  std::map<std::string, std::shared_ptr<System>> systems_;
  real3 cellsize_;
  Grid mastergrid_;
};