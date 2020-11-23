#pragma once

#include <map>
#include <memory>

#include "datatypes.hpp"
#include "grid.hpp"
#include "system.hpp"

class World {
 public:
  /** Create a world with a given cell size and master grid
   *  If the mastergrid has a zero size, then the mastergrid is considered to be
   *  infenitely large.
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

  /** Add a system to this world.
   *  The world will take over ownership of the system.
   */
  void addSystem(std::unique_ptr<System>);

  /** Get a pointer to a system in this world by its name. */
  System* getSystem(std::string name);

 protected:
  std::map<std::string, std::unique_ptr<System>> systems_;
  real3 cellsize_;
  Grid mastergrid_;
};