#pragma once

#include <memory>

#include "datatypes.hpp"
#include "grid.hpp"

class TimeSolver;

class World {
 public:
  /** Create a world with a given cell size and master grid.
   *  If the mastergrid has a zero size, then the mastergrid is considered
   * to be infinitely large.
   */
  explicit World(real3 cellsize, Grid mastergrid = Grid(int3{0, 0, 0}));

  /** Destroy the world. */
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

  /** Return a pointer to the world's timesolver. */
  TimeSolver* timesolver() const;

 protected:
  real3 cellsize_;
  Grid mastergrid_;
  std::unique_ptr<TimeSolver> timesolver_;
};
