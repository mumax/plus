#pragma once

#include <memory>

#include "datatypes.hpp"
#include "grid.hpp"

class TimeSolver;

class World {
 public:
  /** Create a world with a given cell size, mastergrid and pbcRepetitions.
   *
   * @param mastergrid Mastergrid defines a periodic simulation box. If it has
   * zero size in a direction, then it is considered to be infinitely large
   * (no periodicity) in that direction.
   *
   * @param pbcRepetitions The number of repetitions for everything inside
   * mastergrid in the x, y and z directions to create periodic boundary
   * conditions. The number of repetitions determines the cutoff range for the
   * demagnetization.
   * For example {2,0,1} means that, for the strayFieldKernel computation,
   * all magnets are essentially copied twice to the right, twice to the left,
   * but not in the y direction. That row is then copied once up and once down,
   * creating a 5x1x3 grid.
   *
   * @throws std::invalid_argument Thrown when given a negative number of
   * repetitions.
   * @throws std::invalid_argument Thrown when 0 in mastergrid size does not
   * correspond to a 0 in pbcRepetitions.
   */
  explicit World(real3 cellsize, Grid mastergrid, int3 pbcRepetitions);
  /** Create a world with a given cell size and no periodic boundary conditions.
   */
  explicit World(real3 cellsize);

  /** Destroy the world. */
  ~World();

  /** Return the current time of the World */
  real time() const;

  /** Return the cell size dimensions. */
  real3 cellsize() const;

  /** Return the cell volume. */
  real cellVolume() const;

  /** Return the master grid of the world. */
  Grid mastergrid() const;

  /** Returns true if the grid is completely inside the mastergrid. */
  bool inMastergrid(Grid) const;

  /** Return the pbcRepetitions of the world. */
  const int3 pbcRepetitions() const;

  /** Check number of pbcRepetitions.
   * @throws std::invalid_argument Thrown when given a negative number of
   * repetitions.
   */
  void checkPbcRepetitions(const int3 pbcRepetitions) const;

  /** Check arguments of periodic boundary conditions.
   * @throws std:invalid_argument Thrown when 0 in mastergrid does not match 0
   * in pbcRepetitions.
   */
  void checkPbcCompatibility(const Grid mastergrid,
                             const int3 pbcRepetitions) const;

  /** Return a reference to the world's timesolver. */
  TimeSolver& timesolver() const;

 protected:
  real3 cellsize_;
  Grid mastergrid_;
  int3 pbcRepetitions_;
  std::unique_ptr<TimeSolver> timesolver_;
};
