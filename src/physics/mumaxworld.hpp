#pragma once

#include <map>
#include <memory>
#include <stdexcept>

#include "datatypes.hpp"
#include "grid.hpp"
#include "world.hpp"

class Ferromagnet;

/** MumaxWorld is World with additional functionalities for the actual physics
 *  of mumax5.
 */
class MumaxWorld : public World {
 public:
  /** Construct a mumax world. */
  MumaxWorld(real3 cellsize, Grid mastergrid = Grid(int3{0, 0, 0}));

  /** Destroy the world and all systems it contains. */
  ~MumaxWorld();

  /** Uniform bias magnetic field which will affect all magnets in the world. */
  real3 biasMagneticField;

  /** Add a ferromagnet to the world. */
  Ferromagnet* addFerromagnet(Grid grid, std::string name = "");

  /** Get a ferromagnet by its name.
   *  Return a nullptr if there is no magnet with specified name.
   */
  Ferromagnet* getFerromagnet(std::string name) const;

  /** Get map of all Ferromagnets in this world. */
  const std::map<std::string, std::shared_ptr<Ferromagnet>>& ferromagnets()
      const;

 private:
  std::map<std::string, std::unique_ptr<Ferromagnet>> ferromagnets_;
};