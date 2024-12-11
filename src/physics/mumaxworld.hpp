#pragma once

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include "datatypes.hpp"
#include "quantityevaluator.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "torque.hpp"
#include "world.hpp"

class Antiferromagnet;
class Ferromagnet;
class Magnet;
class TimeSolver;

typedef std::function<FM_FieldQuantity(const Ferromagnet*)> FM_Field;

/** MumaxWorld is World with additional functionalities for the actual physics
 *  of mumax⁺.
 */
class MumaxWorld : public World {
 public:
  /** Construct a mumax world. */
  explicit MumaxWorld(real3 cellsize);
  explicit MumaxWorld(real3 cellsize, Grid mastergrid, int3 pbcRepetitions);

  /** Destroy the world and all systems it contains. */
  ~MumaxWorld();

  /** Uniform bias magnetic field which will affect all magnets in the world. */
  real3 biasMagneticField;

  void checkAddibility(Grid grid, std::string name) const;

  /** Add a ferromagnet to the world. */
  Ferromagnet* addFerromagnet(Grid grid,
                              GpuBuffer<bool> geometry,
                              GpuBuffer<unsigned int> regions,
                              std::string name = "");

  /** Add an antiferromagnet to the world. */
  Antiferromagnet* addAntiferromagnet(Grid grid,
                                      GpuBuffer<bool> geometry,
                                      GpuBuffer<unsigned int> regions,
                                      std::string name = "");

    
  /**Add the magnetic field of the other magnets in the new magnet, and vice versa. */
  void handleNewStrayfield(Magnet* newMagnet);

  /** Get a magnet by its name.
   *  Return a nullptr if there is no magnet with specified name. */
  Magnet* getMagnet(std::string name) const;
  /** Get a ferromagnet by its name.
   *  Return a nullptr if there is no ferromagnet with specified name. */
  Ferromagnet* getFerromagnet(std::string name) const;
  /** Get an antiferromagnet by its name.
   *  Return a nullptr if there is no antiferromagnet with specified name. */
  Antiferromagnet* getAntiferromagnet(std::string name) const;


  /** Get map of all Magnets in this world. */
  const std::map<std::string, Magnet*> magnets() const;
  /** Get map of all Ferromagnets in this world. */
  const std::map<std::string, Ferromagnet*> ferromagnets() const;
  /** Get map of all Antiferromagnets in this world. */
  const std::map<std::string, Antiferromagnet*> antiferromagnets() const;

  /** Minimize the current energy state of the world with every magnet in it. */
  void minimize(real tol = 1e-6, int nSamples = 10);
  /** Relax the current state of the world with every magnet in it. */
  void relax(real tol);
  real RelaxTorqueThreshold;

  void resetTimeSolverEquations(FM_Field torque = torqueQuantity) const;


  // --------------------------------------------------
  // PBC

  /** Check if all magnets fit inside the given grid.
   * @throws std:out_of_range Thrown not all magnets fit inside the given grid.
   */
  void checkAllMagnetsInMastergrid() const;

  /** Recalculate the kernels of all strayfields of all magnets in the world. */
  void recalculateStrayFields();

  /** Returns Grid which is the minimum bounding box of all magnets currently
   * in the world.
   * 
   * @throws std::out_of_range Thrown if there are no magnets in the world.
   */
  Grid boundingGrid() const;

  /** Set the periodic boundary conditions.
   * 
   * This will recalculate all strayfield kernels of all magnets in the world.
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
   * @throws std:out_of_range Thrown not all magnets fit inside the given grid.
   */
  void setPBC(const Grid mastergrid, const int3 pbcRepetitions);

  /** Set the periodic boundary conditions
   * 
   * The mastergrid will be set to the minimum bounding box of the magnets
   * currently inside the world, but infinitely large (size 0, no periodicity)
   * for any direction set to 0 in `pbcRepetitions`.
   * 
   * This will recalculate all strayfield kernels of all magnets in the world.
   * 
   * This function reflects the behavior of the mumax³ SetPBC function.
   * 
   * @param pbcRepetitions The number of repetitions for everything inside
   * mastergrid in the x, y and z directions to create periodic boundary
   * conditions. The number of repetitions determines the cutoff range for the
   * demagnetization.
   * 
   * @throws std::invalid_argument Thrown when given a negative number of
   * repetitions.
   * @throws std::out_of_range Thrown if there are no magnets in the world.
   */
  void setPBC(const int3 pbcRepetitions);

  /** Change pbcRepetitions of the world.
   * This does not change the `mastergrid`.
   * 
   * @param pbcRepetitions The number of repetitions for everything inside
   * mastergrid in the x, y and z directions to create periodic boundary
   * conditions. The number of repetitions determines the cutoff range for the
   * demagnetization.
   * 
   * For example {2,0,1} means that, for the strayFieldKernel computation,
   * all magnets are essentially copied twice to the right, twice to the left,
   * but not in the y direction. That row is then copied once up and once down,
   * creating a 5x1x3 grid.
   *
   * This will recalculate all strayfield kernels of all magnets in the world.
   * 
   * @throws std::invalid_argument Thrown when given a negative number of
   * repetitions.
   * @throws std::invalid_argument Thrown when 0 in mastergrid size does not
   * correspond to a 0 in pbcRepetitions.
   */
  void setPbcRepetitions(const int3 pbcRepetitions);

  /** Change the master grid of the world.
   * This does not change the `pbcRepetitions`.
   * 
   * @param mastergrid defines a periodic simulation box. If it has zero size in
   * a direction, then it is considered to be infinitely large (no periodicity) in
   * that direction.
   * 
   * This will recalculate all strayfield kernels of all magnets in the world.
   * 
   * @throws std::invalid_argument Thrown when 0 in mastergrid size does not
   * correspond to a 0 in pbcRepetitions.
   * @throws std:out_of_range Thrown not all magnets fit inside the given grid.
   */
  void setMastergrid(const Grid mastergrid);

  /** Remove the periodic boundary conditions.
   * This will recalculate all strayfield kernels of all magnets in the world.
   */
  void unsetPBC();

  // --------------------------------------------------


 private:
  std::map<std::string, Magnet*> magnets_;
  std::map<std::string, std::unique_ptr<Ferromagnet>> ferromagnets_;
  std::map<std::string, std::unique_ptr<Antiferromagnet>> antiferromagnets_;
};
